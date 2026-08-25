# -*- coding: UTF-8 -*-
"""模型加载、校准、WikiText2 PPL 评测。

要点：
- 单卡加载（不用 device_map="auto"），否则 8B 会被拆到 npu:0/npu:1，
  量化时报 "Expected all tensors on the same device"；
- test_ppl 与 ofmr 示例算法一致（滑窗、seqlen 分块、交叉熵累积 → exp 平均 NLL）。
"""
import time

import torch
import torch_npu  # noqa: F401  必须导入以启用 .npu()
import torch.nn as nn
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model(model_path, seqlen=2048, dtype=torch.float16, device="npu:0",
              device_map=None):
    """加载模型 + tokenizer。dtype 默认 float16（与已验证 baseline 一致）。

    - device_map=None：单卡加载后整体 .to(device)（8B 走这条，已验证）；
    - device_map="auto"/dict：accelerate 跨卡切分（35B 单卡放不下，走这条）。
      注意 amct 换层后 offload 到 CPU/disk 的权重不会被加载 → 强制要求全放 NPU，
      出现 cpu/disk offload 直接报错（提示加卡或降 dtype）。

    ⚠️安全：qwen3_5_moe 等自定义架构必须 trust_remote_code=True，加载时会执行模型仓内的
    modeling_*.py / tokenization_*.py（等同以当前进程权限运行任意 Python 代码）。因此
    **只能把 model_path 指向受信任来源**（官方 ModelScope / HuggingFace 仓，最好固定到已知安全
    的版本/commit hash）；切勿指向来路不明或可被篡改的模型目录，否则会导致主机被控/数据泄露。
    """
    print(f"Loading model: {model_path} (seqlen={seqlen}, dtype={dtype}, "
          f"device_map={device_map})")
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, device_map=device_map, trust_remote_code=True
        )
        dmap = getattr(model, "hf_device_map", {}) or {}
        offloaded = sorted({str(v) for v in dmap.values() if str(v) in ("cpu", "disk")})
        if offloaded:
            raise RuntimeError(
                f"device_map 出现 {offloaded} offload：显存不足，amct 换层后无法加载被 "
                f"offload 的权重。请增加可见卡数（ASCEND_RT_VISIBLE_DEVICES）或降低精度。")
        used = sorted({str(v) for v in dmap.values()})
        print(f"[device_map] 跨 {len(used)} 卡：{used}")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        model = model.to(device)
    model.seqlen = seqlen
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    return model, tokenizer


def input_device(model):
    """模型输入应送达的设备：input embedding 所在卡（多卡时未必是 model.device）。"""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return getattr(model, "device", "npu:0")


def report_coverage_memory(model, skip_layers=("lm_head",),
                           quant_bits=4.25, bf16_bits=16):
    """统计量化层覆盖率与权重显存降低（在 amct.quantize 之前调用，此时 Linear 完整）。

    - 覆盖率 = 被量化的 nn.Linear 数 / 全部 nn.Linear 数（skip_layers 不量化）。
    - 显存：MXFP4 权重 4.25 bit/param（4bit 元素 + E8M0 8bit/32 组），对比 bf16 16 bit。
    """
    n_total = n_quant = 0
    w_all = w_quant = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            n_total += 1
            wp = m.weight.numel()
            w_all += wp
            if not any(s in name for s in skip_layers):
                n_quant += 1
                w_quant += wp
    cov = 100.0 * n_quant / max(n_total, 1)
    orig = w_all * bf16_bits
    new = (w_all - w_quant) * bf16_bits + w_quant * quant_bits
    red = 100.0 * (1 - new / orig)
    print(f"[coverage] 量化 nn.Linear: {n_quant}/{n_total} = {cov:.1f}%  (skip={list(skip_layers)})")
    print(f"[memory] Linear 权重 {bf16_bits}→{quant_bits} bit；整体权重显存降低 = {red:.1f}%")
    return cov, red


@torch.no_grad()
def report_actual_coverage(model, quant_bits=4.25, bf16_bits=16):
    """量化+校准之后复核【真实】覆盖率与显存降低。

    MoE 冷 expert：校准前向从未被路由到 → 其量化模块 forward 从未触发（cur_batch==0）
    → 权重仍是 fp16。量化前的理论覆盖率会把它算进去、虚高。这里按每个量化模块的
    cur_batch 判定它到底有没有被量化（batch_num=1 时，被量化过的 cur_batch>=2）。
    """
    n_quant_mod = n_done = 0
    w_all = w_done = 0
    cold = []
    for name, m in model.named_modules():
        # amct 量化算子（GPTQuant/MRGPTQuant）具备 cur_batch + wts_type
        if hasattr(m, "cur_batch") and hasattr(m, "wts_type") and hasattr(m, "weight"):
            n_quant_mod += 1
            wp = m.weight.numel()
            w_all += wp
            if getattr(m, "cur_batch", 0) >= 2:   # 触发过 get_opt → 已量化
                n_done += 1
                w_done += wp
            else:
                cold.append(name)
    if n_quant_mod == 0:
        return None
    cov = 100.0 * n_done / n_quant_mod
    # 显存：仅真实量化的权重按 4.25bit，冷 expert 仍按 bf16
    orig = w_all * bf16_bits
    new = (w_all - w_done) * bf16_bits + w_done * quant_bits
    red = 100.0 * (1 - new / orig)
    print(f"[actual-coverage] 实际量化模块: {n_done}/{n_quant_mod} = {cov:.1f}%"
          f"（冷 expert 未量化: {len(cold)}）")
    print(f"[actual-memory] 量化权重占比对应显存降低 = {red:.1f}%")
    if cold:
        print(f"[actual-coverage] 冷 expert 示例: {cold[:5]}")
    return cov, red


@torch.no_grad()
def report_memory_full(model, quant_bits=4.25, bf16_bits=16):
    """按【全模型参数】口径算真实显存降低（验收硬指标"显存降低≥50%"）。

    分母 = 所有参数（含 embedding/norm/router 等 fp16）；分子 = 真正被量化的参数：
      ① 量化的 nn.Linear（amct 量化模块，cur_batch>=2）；
      ② 融合 MoE 路由专家（本项目手动量化）。
    对 8B dense 无②，退化为纯 nn.Linear 口径。

    ⚠️ 两类分子都要求【量化确已发生的证据】，不按模块类型名推定：
      ① 看 `cur_batch >= 2`（amct 量化模块走过校准前向）；
      ② 看 `quantize_fused_experts` 写下的 `_mrgptq_fused_quantized` 标记——
         该函数按 `min_experts_numel` 会跳过小张量，且可能整个未被调用；
         若按类型名无条件计入，这两种情况下占比与显存降低都会虚高。
    """
    quant_ids = set()
    fused_seen = fused_marked = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ == "Qwen3_5MoeExperts":
            fused_seen += 1
            marked = getattr(mod, "_mrgptq_fused_quantized", None)
            if marked is None:
                continue          # 未量化：不计入分子
            fused_marked += 1
            for pn in marked:
                p = getattr(mod, pn, None)
                if p is not None:
                    quant_ids.add(id(p))
        elif (hasattr(mod, "cur_batch") and hasattr(mod, "wts_type")
              and hasattr(mod, "weight") and getattr(mod, "cur_batch", 0) >= 2):
            quant_ids.add(id(mod.weight))

    if fused_seen and fused_marked < fused_seen:
        print(f"[memory-full] ⚠️ {fused_seen} 个融合专家模块中仅 {fused_marked} 个带量化标记，"
              f"其余按未量化计入分母")

    total = quant = 0
    for p in model.parameters():
        total += p.numel()
        if id(p) in quant_ids:
            quant += p.numel()
    red = 100.0 * (1 - (quant * quant_bits + (total - quant) * bf16_bits)
                   / (total * bf16_bits))
    print(f"[memory-full] 全模型量化参数占比 = {100.0 * quant / max(total, 1):.1f}% "
          f"({quant / 1e9:.2f}B / {total / 1e9:.2f}B)")
    print(f"[memory-full] 整体权重显存降低 = {red:.1f}%  (要求≥50%)")
    return red


@torch.no_grad()
def calibrate(model, calib_blocks):
    """喂一次校准前向：把校准块拼成一个 batch 过模型。

    对 GPTQ：该次前向内累积 Hessian 并当场做 Cholesky 误差补偿（batch_num=1）。
    calib_blocks: list[[1, block_size]] → 堆成 [n_blocks, block_size]。

    ⚠️ 校准集为空是**静默**故障：Hessian 全零时 Cholesky 未必报错（有阻尼兜底），
    补偿退化为无操作，只表现为 PPL 变差、查不到源头，故显式拦截。

    注：`--calib_block`（默认 512）与评测 `--seq_len` 相互独立，校准块**短于** seqlen
    是正常工况（下面的 `calib[:, :seqlen]` 此时是恒等操作）。只有块**长于** seqlen 时
    才会发生截断，此时按截断前长度算 token 数会高报，故 n_tok 取二者较小值。
    """
    if not calib_blocks:
        raise ValueError(
            "校准集为空：calib_blocks 没有任何块。GPTQ 必须有校准数据才能累积 Hessian，"
            "空输入会让误差补偿退化为无操作。请检查 --n_calib / --calib_block 与数据集加载。")
    calib = torch.cat(calib_blocks, dim=0)
    blk_len = calib.shape[1]
    if blk_len > model.seqlen:
        print(f"[calib] 注：校准块长 {blk_len} > seq_len {model.seqlen}，"
              f"每块将被截断到 {model.seqlen} 个 token 送入")
    n_tok = calib.shape[0] * min(blk_len, model.seqlen)
    # Hessian H=2XXᵀ 是 [in,in]，样本数少于 in_features 时秩亏、补偿等于对校准集过拟合。
    # 全网最大的 in_features（Qwen3-8B 的 down_proj=12288）决定了所需样本量的下限。
    # ⚠️本函数在 amct.quantize() 之后调用，nn.Linear 已被换成量化模块（非 nn.Linear
    # 子类）——按 isinstance(nn.Linear) 扫只剩 lm_head，会把 in_features 报小。
    # 改为扫所有持 2D weight 的模块，取 weight.shape[1]。
    max_in = 0
    for m in model.modules():
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2:
            max_in = max(max_in, w.shape[1])
    ratio = n_tok / max_in if max_in else float("inf")
    print(f"[calib] shape={tuple(calib.shape)}  tokens={n_tok}  "
          f"最大 in_features={max_in}  样本/维度比={ratio:.1f}:1")
    if ratio < 4:
        # 该比值仅作诊断，**不代表需要调高**。本项目实测（Qwen3-8B, W4A4）：
        #   --n_calib 64→256（比值 1.1:1→5.2:1），全量 73 块 delta 0.4463→0.4520，
        #   即无收益、略劣。20 块子集上曾显示 0.048 的"收益"，实为子集偏差假象
        #   （同一对配置的 20/73 偏移在 −0.011 与 +0.042 之间摆动，摆幅 0.053）。
        # 结论：Hessian 估计不足在理论上成立，但本设置下不是精度瓶颈。
        print(f"[calib] 注：样本/维度比 {ratio:.1f}:1（GPTQ 文献常用 ≥20:1）。"
              f"本项目实测提高到 5.2:1 在全量评测上未见收益（详见报告 §4.3），"
              f"此处仅作诊断输出。--n_calib 计的是语料【条数】不是块数")
    t0 = time.time()
    ids = calib[:, : model.seqlen].to(input_device(model))
    # 只跑 transformer 主体，**不过 lm_head**：校准只需要各层 Hessian 与补偿，
    # logits 算完即弃，而它是 [n_blocks, seq, vocab]——128k token × 151936 词表
    # ≈ 36GiB，是全流程最大的单个张量，且纯属浪费（lm_head 本就在 skip_layers 里）。
    # use_cache=False：校准是一次性整段前向、无增量解码，KV cache 建完即弃。
    # Qwen3-8B(GQA, 8kv×128) 每层 K+V = 2·N·1024·2B，36 层在 128k token 下是 18.9GB
    # 的纯浪费（256k token 时 37.7GB）。关掉不影响任何数值——缓存只服务增量解码。
    body = getattr(model, "model", None)
    if callable(body):
        body(ids, use_cache=False)
    else:  # 兜底：模型结构不含 .model 时退回整模型前向
        model(ids, use_cache=False)
    torch_npu.npu.empty_cache()
    print(f"[calib] done in {time.time() - t0:.1f}s")


@torch.no_grad()
def test_ppl(model, testenc, max_samples=0):
    """WikiText2 PPL。testenc: [1, total_tokens] 的 input_ids（已在目标 device）。
    max_samples>0 时只评前 N 块（消融/快速验证用）。"""
    seqlen = model.seqlen
    dev = input_device(model)
    nsamples = testenc.numel() // seqlen
    if max_samples and 0 < max_samples < nsamples:
        nsamples = max_samples                 # 只评前 N 块（消融/快速验证用；PPL 略噪但定性可靠）
        print(f"[ppl] 仅评前 {nsamples} 块（--max_eval_samples）")
    nlls = []
    t0 = time.time()
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, i * seqlen : (i + 1) * seqlen].to(dev)
        logits = model(batch, use_cache=False).logits  # 逐块整段前向，KV cache 无用
        shift_logits = logits[:, :-1, :].contiguous().float()
        # 多卡时 lm_head 输出可能落在另一张卡 → labels 对齐到 logits 所在设备
        shift_labels = testenc[:, i * seqlen : (i + 1) * seqlen][:, 1:].to(shift_logits.device)
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nlls.append(loss.float() * seqlen)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    dt = time.time() - t0
    print(f"Test time: {dt // 60:.0f}min {dt % 60:.1f}s | nsamples={nsamples}")
    print(f"Score: {ppl.item()}")
    return ppl.item()
