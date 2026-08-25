# -*- coding: UTF-8 -*-
"""MR-GPTQ（Micro-Rotated GPTQ, arXiv:2509.23202）MXFP4 实现。

子类化 amct 的 GPTQuant，override 权重优化，做三件事：
  1. **标准 block-32 MXFP4 量化**（沿 in-features、每 32 元素共享 E8M0 指数），
     取代 stock 的 per-column `.repeat(1,2)` 非标准缩放；
  2. **shared exponent 预计算一次 + CPU 上做补偿**——消除 stock 每列重算 20+ 小算子、
     百万次 NPU kernel 启动的开销（2h → 分钟级）；
  3. MR-GPTQ 的增量（`_INCREMENTS` 开关控制，便于消融）：
       - `hadamard`      block-wise Hadamard 微旋转
       - `scale_fitting` MXFP scale fitting（论文附录 H）
     论文的第三项增量「静态激活重排」**有意未实现**——本实现可对每层施加全宽旋转
     （`_HADAMARD_FULL`），旋转后各通道近似可交换，按幅度分组已无物可分。
     （GPTQ 自带的 act-order，即按 Hessian 对角降序处理列，在补偿主循环中始终启用，
     与论文那项增量是两回事。）
  4. W4A4（评审补充要求）：`act_quant` 及其优化项 `act_scale_fit` / `joint_comp` /
     `_HADAMARD_FULL` / `_PERC_DAMP` / `_ACT_PROTECT`。另有 3 个实测为负结果的开关
     （`act_clip` / `weight_clip` / `act_fit_rowwise`）默认关闭、保留作消融记录。

零改 amct 源码：import 本模块即运行时注册 'mr_gptq'。
"""
import math
import time

import torch
import torch.nn.functional as F

from amct_pytorch.classic.quantize_op.gptq_module import GPTQuant
from amct_pytorch.common.utils.quant_util import cal_shared_exponent
from amct_pytorch.common.utils.data_utils import float_to_fp4e2m1
from amct_pytorch.common.utils.vars import MXFP4_E2M1


# MR-GPTQ 增量开关（默认全关 = GPTQ-mxfp4 baseline；逐个打开做消融）
_INCREMENTS = {
    "hadamard": False,        # block-wise Hadamard 微旋转
    "scale_fitting": False,   # MXFP scale fitting（附录 H），作用于权重
    # 注：论文第三项增量「静态激活重排」有意未实现（理由见模块 docstring 与设计文档 §3.3），
    # 故此处不设开关——留一个永不生效的开关会让日志显示 True 却无任何行为，比没有更糟。
    "act_quant": False,       # W4A4：激活也做 MXFP4（动态、逐32块）；须配 hadamard 压离群值
    "act_scale_fit": False,   # W4A4 优化①：激活量化也用 scale fitting（次幂次细网格，非幂次）
    "joint_comp": False,      # W4A4 优化②：用量化后的激活建 Hessian，权重补偿适配 W4A4 输入
    "act_clip": False,        # W4A4 优化③：激活 scale 裁剪，逐块搜 α 最小化 MSE（❌实测负结果）
    "weight_clip": False,     # W4A4 优化④：权重 scale 裁剪搜索（校准期一次，评测零开销）
    "act_fit_rowwise": False, # W4A4 优化⑤：激活细网格跨度逐 token 拟合（须配 act_scale_fit）
}

# 激活 scale 裁剪候选系数：scale ← α·scale。含 1.0（=不裁剪）→ 逐块取 MSE 最优者时，
# 结果在 MSE 意义上**不可能劣于**不裁剪。跨度取宽（0.6~1.0）而非细密，因最优 α 未知且逐块异质。
_ACT_CLIP_GRID = (1.0, 0.9, 0.8, 0.7, 0.6)

# 权重 scale 裁剪候选：比激活侧更密且更靠近 1.0——权重误差会被 GPTQ 逐列补偿掉一部分，
# 最优裁剪通常较温和；且本搜索只在校准期跑一次，多给候选不增加评测开销。
_WEIGHT_CLIP_GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)

# True：每层旋转块取满 in_features（全宽旋转），不可构造时退回 _HADAMARD_K。
# 唯一受影响的是 in≠_HADAMARD_K 的层——Qwen3-8B 上就是 down_proj(in=12288)。
_HADAMARD_FULL = False

_HADAMARD_K = 128  # block-wise Hadamard 尺寸；论文取 128 是为【在线算子】的开销权衡。
                   # 本项目 fake-quant 只测精度、旋转无推理开销 → 可经 --hadamard_k 调大
                   # (512/1024/4096) 做消融：块越大离群值摊得越开，或能进一步压 W4A4 误差。
                   # 注意 in_features 须被 k 整除，否则该层跳过旋转（见 _do_hadamard）。

# W4A4 混合精度：层名含这些子串的层【不】量化激活（保 bf16/高精度），用于保护敏感层冲 0.4。
# 例：['down_proj'] 保护所有 down_proj 的激活；空 = 全量 W4A4。
_ACT_PROTECT = []

# GPTQ Hessian 阻尼系数覆盖（None = 用 amct 父类内置值）。阻尼过小 Cholesky 易失稳、
# 过大则补偿被削弱，是 GPTQ 少数值得一调的超参。
_PERC_DAMP = None

# 校准前向的分块大小（沿 batch 维，单位=块数；0 = 不分块，父类原行为）。
# 作用于 Hessian 累积与 fake_quant_forward 两条链路，只降峰值显存以放大校准集。
# 评测期输入 dim0=1，不触发。
_CALIB_CHUNK = 0

_ACT_QUANT_SKIPPED = set()  # 因无法旋转而放弃激活量化的层名（去重打印，见 _do_act_quant）

_PROGRESS = [0]  # 已补偿层计数（进度显示用；每进程从 0 开始）

_FP4_MAX_EXP = 2  # E2M1 最大指数（6 = 1.5·2²）；scale fitting 的 offset


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def _sylvester_hadamard(k):
    """归一化 Sylvester-Hadamard H_k/√k（k 须为 2 的幂），正交且对称。"""
    H = torch.ones(1, 1)
    while H.shape[0] < k:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / (k ** 0.5)


# Paley-I 可构造的非 2 幂阶：m = q+1，q 为素数且 q ≡ 3 (mod 4)。
# 本项目只用到 12（= 12288/1024，见 _build_hadamard）；其余留作其它模型维度的兜底。
_PALEY_ORDERS = (12, 20, 24, 44, 48, 60, 68, 72)


def _paley_hadamard(m):
    """Paley I 型构造，阶 m = q+1（q 素数、q ≡ 3 mod 4）。

    S = [[0, 1ᵀ], [−1, Q]]，Q[i,j] = χ(i−j mod q)，χ 为 GF(q) 上的二次剩余特征
    （χ(0)=0，非零平方剩余取 +1，非剩余取 −1）。取 H = I + S，则 S 反对称、
    S·Sᵀ = q·I，于是 H·Hᵀ = (I+S)(I−S) = I − S² = (1+q)·I = m·I。
    注意：H **只保证正交，不对称**——但旋转只需要正交（y = (xH)(WH)ᵀ = x·Hᵀ H·Wᵀ = x·Wᵀ）。
    """
    q = m - 1
    residues = {(i * i) % q for i in range(1, q)}
    chi = [0.0 if a == 0 else (1.0 if a in residues else -1.0) for a in range(q)]
    Q = torch.tensor([[chi[(i - j) % q] for j in range(q)] for i in range(q)])
    S = torch.zeros(m, m)
    S[0, 1:] = 1.0
    S[1:, 0] = -1.0
    S[1:, 1:] = Q
    H = (torch.eye(m) + S) / (m ** 0.5)
    err = (H.matmul(H.t()) - torch.eye(m)).abs().max().item()
    if err > 1e-5:
        raise ValueError(f"Paley-{m} 构造失败：正交性误差 {err:.2e}")
    return H


def _hadamard_order_ok(n):
    """n 阶 Hadamard 是否可构造：2 的幂走 Sylvester，否则找 m·2ᵏ 分解。"""
    if _is_pow2(n):
        return True
    return any(n % m == 0 and _is_pow2(n // m) for m in _PALEY_ORDERS)


def _build_hadamard(n):
    """构造 n 阶归一化 Hadamard。

    n 为 2 的幂 → Sylvester。否则拆成 m·2ᵏ 走 Kronecker 积：正交矩阵的 Kronecker
    积仍正交，故 H_n = H_m ⊗ H_{2ᵏ} 合法。
    例：Qwen3-8B 的 down_proj in_features = 12288，**不是 2 的幂**，Sylvester 造不出来；
    12288 = 3 × 4096 这条路走不通（3 阶 Hadamard 不存在），只能走 12288 = 12 × 1024。
    """
    if _is_pow2(n):
        return _sylvester_hadamard(n)
    for m in _PALEY_ORDERS:
        if n % m == 0 and _is_pow2(n // m):
            return torch.kron(_paley_hadamard(m), _sylvester_hadamard(n // m))
    raise ValueError(f"无法构造 {n} 阶 Hadamard（不是 2 的幂，也无 Paley 分解）")


# True：旋转前乘随机 ±1 对角阵 D（randomized Hadamard，QuaRot / QuIP# 的标准做法）。
# 动机：归一化 Sylvester 的**第 0 行是全 1**，故 (xH)[0] = Σxⱼ/√k = √k·mean(x)。
# 量化激活的几层其输入是 RMSNorm 输出，而 RMSNorm **不做中心化**、均值非零 →
# 旋转反而在坐标 0 上**制造**一个 ~√k 倍于均值的离群值，把它所在的第 0 个 32 块的
# 倍率顶高、连累同块另外 31 个坐标。乘 D 之后全 1 行变随机符号求和，DC 分量被打散。
# 无损：D 对角且 D²=I → (xDH)(WDH)ᵀ = xD·HHᵀ·DWᵀ = xDDWᵀ = xWᵀ。
_HADAMARD_RANDOM = False

_HADAMARD_SIGN_SEED = 0  # D 的随机种子；权重与激活必须用同一个 D，故按 k 固定生成

_HADAMARD_CACHE = {}   # (k, device_str, randomized) -> 归一化 Hadamard（随机化时已折入 D）


def _get_hadamard(k, device="cpu"):
    """按 (k, device, 是否随机化) 缓存。矩阵可以很大（k=12288 → 576MiB fp32），252 个
    量化模块各建一份、或每次前向各搬一份上卡，都会撑爆内存 → 全局共享只读。

    随机化时**把 D 折进矩阵**：(x·D)·H = x·(D·H)，即按 dᵢ 翻转 H 的第 i 行。
    这样旋转仍是一次 matmul，**不引入额外临时张量、不增加计算**——若在 `_rotate_last_dim`
    里现乘 D，会多出一个与输入等大的 fp32 副本（down_proj 上就是 679MB），
    且 `.float()` 在输入已是 fp32 时返回自身、就地乘会踩到别名。
    """
    key = (k, str(device), _HADAMARD_RANDOM)
    H = _HADAMARD_CACHE.get(key)
    if H is None:
        cpu_key = (k, "cpu", _HADAMARD_RANDOM)
        base = _HADAMARD_CACHE.get(cpu_key)
        if base is None:
            base = _build_hadamard(k)
            if _HADAMARD_RANDOM:
                # 固定种子 → 权重与激活拿到同一个 D（否则 (xD₁H)(WD₂H)ᵀ ≠ xWᵀ，结果直接错），
                # 且跨进程可复现。
                g = torch.Generator().manual_seed(_HADAMARD_SIGN_SEED + k)
                d = torch.randint(0, 2, (k,), generator=g, dtype=torch.float32) * 2 - 1
                base = base * d.unsqueeze(1)          # D·H：按 dᵢ 翻转第 i 行
            _HADAMARD_CACHE[cpu_key] = base
        H = base if key == cpu_key else base.to(device)
        _HADAMARD_CACHE[key] = H
    return H


class MRGPTQuant(GPTQuant):
    """MR-GPTQ 量化算子。当前实现 = 高效标准 MXFP4 + GPTQ 误差补偿 + 可选增量。"""

    def __init__(self, ori_module, layer_name, quant_config):
        super().__init__(ori_module, layer_name, quant_config)
        self.increments = dict(_INCREMENTS)
        if self.wts_type != MXFP4_E2M1:
            raise ValueError(f"MR-GPTQ 目前仅支持 mxfp4_e2m1，收到 {self.wts_type}")
        # 旋转块大小：默认全局 _HADAMARD_K；开了 _HADAMARD_FULL 则按本层 in_features
        # 取满（全宽旋转）——k=4096 下 in=12288 的 down_proj 只能切成 3 个对角块，
        # 是全网唯一没拿到全宽旋转的层，而它恰是最大的层。
        self.hadamard_k = _HADAMARD_K
        in_features = self.weight.shape[1]
        if _HADAMARD_FULL and in_features != _HADAMARD_K and _hadamard_order_ok(in_features):
            self.hadamard_k = in_features
        _get_hadamard(self.hadamard_k)  # 预热 CPU 侧缓存（同 k 全局只造一次）
        if _PERC_DAMP is not None:
            self.perc_damp = _PERC_DAMP

    # ---- W4A4：本层是否量化激活（混合精度下敏感层跳过，保高精度） ----------
    def _do_act_quant(self):
        if not self.increments.get("act_quant"):
            return False
        if any(p in self.layer_name for p in _ACT_PROTECT):
            return False
        # ⚠️ W4A4 的前提是**旋转先把激活离群值压平**，4-bit 激活才不崩。若本层因
        # in_features 不被 k 整除而跳过了旋转（见 _do_hadamard），仍量化激活会显著劣化，
        # 且劣化容易被误读为"该模型难量化"。此处兜底：请求了旋转但本层拿不到 → 不量化激活。
        # Qwen3-8B 各层 in ∈ {4096, 12288} 恒可整除，故该分支在 8B 上永不触发、行为不变；
        # 维度较杂的模型（如 qwen3_5_moe 的部分 linear_attn 投影）才会命中。
        # 注意条件用 increments["hadamard"]：全局不开旋转时（W4A4 无旋转消融）不应拦。
        if self.increments["hadamard"] and not self._do_hadamard():
            if self.layer_name not in _ACT_QUANT_SKIPPED:
                _ACT_QUANT_SKIPPED.add(self.layer_name)
                print(f"[MR-GPTQ] ⚠️ {self.layer_name} (in={self.weight.shape[1]}) 不被 "
                      f"k={self.hadamard_k} 整除、无法旋转 → **跳过激活量化**"
                      f"（W4A4 需旋转压离群值）。该层激活保高精度，"
                      f"应计入「非真 4-bit 激活」层数", flush=True)
            return False
        return True

    # ---- block-wise Hadamard 旋转（增量①） --------------------------------
    def _do_hadamard(self):
        # in_features 需被 k 整除；否则该层跳过旋转（Qwen3 各层均可整除）
        return self.increments["hadamard"] and self.weight.shape[1] % self.hadamard_k == 0

    def _rotate_last_dim(self, x):
        """沿最后一维（in）做块对角 Hadamard：reshape [...,n_blk,k] @ H_k。
        matmul 在 fp32 下做再回原 dtype——k 大时（如 4096）fp16 累加 4096 项误差可达数个
        百分点，会污染消融；fp32 旋转对 k=128 影响可忽略、对大 k 是必需。"""
        k = self.hadamard_k
        shape = x.shape
        H = _get_hadamard(k, x.device)  # 按 (k, device) 全局共享，避免每次前向搬一份大矩阵
        xr = x.reshape(*shape[:-1], shape[-1] // k, k).float().matmul(H)
        return xr.reshape(shape).to(x.dtype)

    def update_hessian(self, input_data):
        """分块入口：把【旋转 + 激活量化 + Hessian 累积】整条链路都放进分块循环。

        三步的中间量都随校准 token 线性增长——旋转的 fp32 副本、`_quant_direct` 的
        scale_full/除法/`float_to_fp4e2m1` 内部掩码、父类的 [cin, N] fp32 转置。
        只分块最后一步不够：256k token 下前两步会先 OOM。
        """
        if _CALIB_CHUNK and input_data.dim() == 3 and input_data.shape[0] > _CALIB_CHUNK:
            for chunk in torch.split(input_data, _CALIB_CHUNK, dim=0):
                self._update_hessian_chunk(chunk)
        else:
            self._update_hessian_chunk(input_data)

    def _update_hessian_chunk(self, input_data):
        """单块的旋转 + （联合补偿下的）激活量化 + Hessian 累积。

        等价性：Hessian 累积因父类 running-average 而与整块【严格等价】；旋转是逐 token
        线性变换，同样与分块无关。唯一有差别的是 `act_scale_fit` 的细网格拟合——它取
        全张量的 Lmin/Lmax，分块后变成逐块拟合。这反而更贴近推理期行为（评测时每个
        block 单独前向，本就是逐 block 拟合），故取分块语义。
        """
        # 旋转后累积 Hessian，使 H_r = RᵀHR 与旋转权重一致（转输入比转 Hessian 便宜）
        if self._do_hadamard():
            input_data = self._rotate_last_dim(input_data)
        # W4A4 优化②（联合补偿）：用【量化后】的激活建 Hessian，使权重补偿适配 W4A4 推理输入
        if self.increments.get("joint_comp") and self._do_act_quant():
            input_data = self._quant_act_mxfp4(
                input_data, self.group_size, self.increments.get("act_scale_fit", False),
                self.increments.get("act_clip", False),
                self.increments.get("act_fit_rowwise", False))
        # 父类 running-average（hessian *= n/(n+b); n += b）保证分块累积严格等价：
        #   整块 h=(2/B)·XXᵀ；两块 h=(2/B)(X₁X₁ᵀ+X₂X₂ᵀ) —— 相同。
        super().update_hessian(input_data)

    # ---- MXFP scale fitting（附录 H 式2/3）：E8M0 网格重映射 ----------------
    @staticmethod
    def _fit_block_scale(weight, block_size=32, per_row=False):
        """把每块连续 log-scale 重映射到 [L_min,L_max] 的 256 级细网格，
        得到次幂次(sub-power-of-two) scale，减小 E8M0 粗网格的 scale 量化误差。
        L = log2(block_max) − FP4_MAX_EXP；q=clamp(round(255·(L−Lmin)/(Lmax−Lmin)),0,255)；
        L_fit = Lmin + q·(Lmax−Lmin)/255；返回每块浮点 scale [out, n_groups]。

        per_row=False（默认，**权重侧**）：Lmin/Lmax 取全张量极值。权重分布静态、良性，
            全局网格拟合是论文原意，W4A16 主线结果基于此，勿改。
        per_row=True（**激活侧**，W4A4 优化⑤）：Lmin/Lmax 逐行（逐 token）取。
            激活是 [N_token, in]，全局拟合下**任一 token 上的离群值都会拉宽 256 级网格的
            跨度、降低所有其它 token 的 scale 分辨率**——把静态权重的假设套到动态激活上
            的错配。逐行拟合让每个 token 独享自己的网格跨度。
            注意与 act_clip（负结果）机理不同：那是砍掉块内 max（削弱 attention sink），
            这里 max 分毫不动，只是重新分配 256 个 scale 档位。
        """
        out, in_features = weight.shape
        n_groups = (in_features + block_size - 1) // block_size
        pad = n_groups * block_size - in_features
        w = F.pad(weight, (0, pad)) if pad else weight
        w = w.reshape(out, n_groups, block_size)
        block_max = w.abs().amax(dim=-1)                             # [out, n_groups]
        # 连续理想 log2(scale)：block_max/scale = 6（用满 E2M1 范围，非 4）
        L = torch.log2(block_max) - math.log2(6.0)
        L = torch.where(torch.isfinite(L), L, torch.zeros_like(L))  # 空块→0
        if per_row:
            Lmin = L.amin(dim=-1, keepdim=True)                      # [out, 1] 逐行
            Lmax = L.amax(dim=-1, keepdim=True)
        else:
            Lmin, Lmax = L.min(), L.max()                            # 全局标量
        denom = (Lmax - Lmin).clamp_min(1e-6)
        q = torch.clamp(torch.round(255.0 * (L - Lmin) / denom), 0.0, 255.0)
        L_fit = Lmin + q * (Lmax - Lmin) / 255.0                    # 细网格
        return torch.pow(2.0, L_fit)                                # 每块浮点 scale

    # ---- block-32 MXFP4 单列量化（按每块浮点 scale） ----------------------
    @staticmethod
    def _quant_col(w, scale_col):
        """对单列 w[out] 用其组 scale scale_col[out] 做 MXFP4 fake-quant：
        w/scale → round 到 E2M1 网格 → ×scale。"""
        return float_to_fp4e2m1(w / scale_col) * scale_col

    # ---- 直接 MXFP4 量化（无补偿，RTN） -----------------------------------
    @staticmethod
    def _quant_direct(weight, blk_scale, group_size):
        """整块权重直接 MXFP4 量化：每列按其组 scale round 到 E2M1，不做 GPTQ 补偿。
        用于 MoE 冷/稀疏 expert（Hessian 缺失或秩亏，补偿不可用）——退化为 RTN 好过
        把权重列置零。weight[out,in]，blk_scale[out, n_groups]。"""
        in_features = weight.shape[1]
        scale_full = blk_scale.repeat_interleave(group_size, dim=1)[:, :in_features]
        return float_to_fp4e2m1(weight / scale_full) * scale_full

    # ---- 激活 scale 裁剪搜索（W4A4 优化③） ---------------------------------
    @classmethod
    def _quant_clip_search(cls, x2, blk_scale, group_size, grid=_ACT_CLIP_GRID):
        """逐块搜裁剪系数 α：scale ← α·scale，取块内量化 MSE 最小的 α。

        动机：默认 scale 让块内 |max| 恰好落到 E2M1 上界 6，即**最大的那个元素独占了全部
        动态范围**；E2M1 仅 16 级，其余 31 个元素被挤在粗网格上。适度裁剪（牺牲长尾的
        精确度）能显著提高块内多数元素的分辨率——这是不引入任何高精度层、纯 W4A4 的优化。

        grid 含 1.0 → 每块取 argmin MSE 时，**结果在 MSE 意义上不劣于不裁剪**（最坏退化为原行为）。
        x2[N, C]、blk_scale[N, n_groups]；返回 [N, C]。
        """
        n_rows, columns = x2.shape
        n_groups = blk_scale.shape[1]
        pad = n_groups * group_size - columns
        xg = (F.pad(x2, (0, pad)) if pad else x2).reshape(n_rows, n_groups, group_size)

        best_q = best_err = None
        for alpha in grid:
            s = (blk_scale * alpha).clamp_min(1e-20).unsqueeze(-1)   # [N, G, 1]
            q = float_to_fp4e2m1(xg / s) * s
            err = (q - xg).pow(2).sum(dim=-1, keepdim=True)          # [N, G, 1] 逐块 MSE
            if best_q is None:
                best_q, best_err = q, err
            else:
                take = err < best_err
                best_q = torch.where(take, q, best_q)
                best_err = torch.where(take, err, best_err)
        out = best_q.reshape(n_rows, n_groups * group_size)
        return out[:, :columns] if pad else out

    # ---- 权重 scale 裁剪搜索（W4A4 优化④） ---------------------------------
    @classmethod
    def _search_block_clip(cls, w, blk_scale, group_size, grid=_WEIGHT_CLIP_GRID):
        """逐块搜权重 scale 的裁剪系数 α，**返回最优 scale**（而非量化值）。

        与激活侧 `_quant_clip_search` 的两点关键区别：
        1. **返回 scale**：选出的 scale 要交给 GPTQ 逐列补偿循环复用（`_quant_col`），
           使裁剪与误差补偿协同——补偿会吸收一部分裁剪带来的偏差；
        2. **机理不同故结论不同**：激活侧裁剪为负结果，因被裁的大激活承担 attention sink、
           是模型的功能零件；权重无此性质，且权重误差可被 GPTQ 补偿，裁剪+补偿是
           GPTQ/AWQ 的标准组合。默认 scale 让块内 |max| 恰好落在 E2M1 上界 6，
           即最大元素独占动态范围、其余 31 个元素被挤在粗网格上。

        仅在校准期（`get_opt_weight_and_quant_factor`，CPU）执行一次，**评测期零开销**。
        候选含 1.0 → 最坏退化为不裁剪。w[out, in]、blk_scale[out, n_groups]。
        """
        out, columns = w.shape
        n_groups = blk_scale.shape[1]
        pad = n_groups * group_size - columns
        wg = (F.pad(w, (0, pad)) if pad else w).reshape(out, n_groups, group_size)

        best_scale = best_err = None
        for alpha in grid:
            s = (blk_scale * alpha).clamp_min(1e-20)              # [out, G]
            q = float_to_fp4e2m1(wg / s.unsqueeze(-1)) * s.unsqueeze(-1)
            err = (q - wg).pow(2).sum(dim=-1)                     # [out, G] 逐块 MSE
            if best_scale is None:
                best_scale, best_err = s, err
            else:
                take = err < best_err
                best_scale = torch.where(take, s, best_scale)
                best_err = torch.where(take, err, best_err)
        return best_scale

    # ---- 激活 MXFP4 动态量化（W4A4，增量④） ---------------------------------
    @classmethod
    def _quant_act_mxfp4(cls, x, group_size=32, scale_fit=False, clip=False,
                         fit_rowwise=False):
        """激活 MXFP4 动态伪量化：把 [*, in] 展平成 [N, in]，沿最后一维每 32 通道一块，
        **运行时**算 scale（激活是动态的，每次前向重算）→ 吸附到 E2M1。复用 _quant_direct。
        scale_fit=False：E8M0 幂次 scale；True：附录 H 次幂次细网格（W4A4 优化①，减小激活 scale 误差）。
        clip=True：在此 scale 上再逐块搜裁剪系数（W4A4 优化③，❌负结果）。
        fit_rowwise=True：细网格跨度逐 token 拟合而非全张量（W4A4 优化⑤，见 _fit_block_scale）。
        须先做 Hadamard 旋转压激活离群值，否则 4-bit 激活会崩。"""
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).float()
        if scale_fit:
            blk = cls._fit_block_scale(x2, group_size, per_row=fit_rowwise)  # 次幂次细网格
        else:
            blk = torch.pow(2.0, torch.nan_to_num(cal_shared_exponent(x2)))  # E8M0 幂次
        if clip:
            xq = cls._quant_clip_search(x2, blk, group_size)
        else:
            xq = cls._quant_direct(x2, blk, group_size)
        return xq.reshape(shape).to(x.dtype)

    # ---- Hessian 逆的上三角 Cholesky（override 父类，全程 CPU） -------------
    def cal_hessian_inverse(self, hessian, columns):
        # 父类用 self.weight.device(NPU) 造 index 张量，与 CPU hessian 冲突；
        # 这里全程 CPU：damping → cholesky → 逆 → 上三角 cholesky。
        damp = self.perc_damp * torch.mean(torch.diag(hessian))
        idx = torch.arange(columns)
        hessian[idx, idx] += damp
        L = torch.linalg.cholesky(hessian)
        hinv = torch.cholesky_inverse(L)
        return torch.linalg.cholesky(hinv, upper=True)

    # ---- GPTQ 误差补偿（CPU、标准 MXFP4、shared exponent 预计算） ----------
    @torch.no_grad()
    def get_opt_weight_and_quant_factor(self):
        columns = self.weight.shape[1]
        orig_device = self.weight.device
        orig_dtype = self.weight.dtype

        _PROGRESS[0] += 1
        _t0 = time.time()
        print(f"[MR-GPTQ] layer #{_PROGRESS[0]:>3}  {self.layer_name}  "
              f"(in={columns}, had={self._do_hadamard()}, "
              f"scale_fit={self.increments['scale_fitting']})", flush=True)

        # 全程 CPU float32：小算子无 NPU 启动开销，Cholesky 亦在 CPU（~0.7s）
        weight = self.weight.detach().to("cpu", torch.float32)
        if self._do_hadamard():
            weight = self._rotate_last_dim(weight)  # 旋转权重（Hessian 已在 update 时旋转）

        # 1) 预计算每块浮点 scale（沿 in-features，32 一组）——基于真实权重，置零前算。
        #    形状 [out, ceil(in/32)]；后续每列按 原始列号//32 取用。
        #    增量②：scale fitting → E8M0 网格重映射（附录 H 式2/3）
        if self.increments["scale_fitting"]:
            blk_scale = self._fit_block_scale(weight, self.group_size)
        else:
            blk_scale = torch.pow(2.0, torch.nan_to_num(cal_shared_exponent(weight)))
        # 增量④：权重 scale 裁剪搜索。放在此处 → 选出的 scale 同时供下方 GPTQ 逐列补偿
        # 与各兜底分支的直接量化复用，裁剪与补偿协同。
        if self.increments.get("weight_clip"):
            blk_scale = self._search_block_clip(weight, blk_scale, self.group_size)

        # 2) MoE 冷 expert 兜底：校准前向从未路由到它 → 无 Hessian。
        #    （理论上 get_opt 只在 ≥1 次 update 后触发、hessian 不为 None；防御性保留）
        if self.hessian is None:
            wq = self._quant_direct(weight, blk_scale, self.group_size)
            print(f"[MR-GPTQ]   ↳ no Hessian (cold expert) → direct MXFP4 "
                  f"in {time.time() - _t0:.1f}s", flush=True)
            return wq.to(orig_device, orig_dtype), None, None

        hessian = self.hessian.to("cpu", torch.float32)
        diag = torch.diag(hessian)
        dead = diag == 0

        # 3) MoE 稀疏 expert 兜底：多数输入通道在校准集内零激活 → Hessian 秩亏，
        #    GPTQ 补偿不稳且 weight[:,dead]=0 会毁掉整块权重。退化为直接 MXFP4（RTN）。
        if bool(dead.all()) or dead.float().mean().item() > 0.5:
            wq = self._quant_direct(weight, blk_scale, self.group_size)
            print(f"[MR-GPTQ]   ↳ sparse expert (dead={dead.float().mean().item():.0%}) "
                  f"→ direct MXFP4 in {time.time() - _t0:.1f}s", flush=True)
            return wq.to(orig_device, orig_dtype), None, None

        # 4) 少量死列（dense 层常态）：置 1 稳定 Cholesky、对应权重列置零（stock GPTQ）
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        # 5) 激活重排：按 Hessian 对角（激活能量）排序处理列。
        #    stock GPTQ 亦如此；MR-GPTQ 的“静态激活重排”增量将在此细化。
        perm = torch.argsort(diag, descending=True)
        invperm = torch.argsort(perm)
        weight = weight[:, perm]
        hessian = hessian[perm][:, perm]

        # 6) Hessian 逆的上三角 Cholesky（override 版，全程 CPU）
        #    极端秩亏时仍可能非正定 → 兜底直接 MXFP4（用旋转后原权重，未置零）。
        try:
            hessian_inv = self.cal_hessian_inverse(hessian, columns)
        except torch.linalg.LinAlgError:
            w0 = self.weight.detach().to("cpu", torch.float32)
            if self._do_hadamard():
                w0 = self._rotate_last_dim(w0)
            wq = self._quant_direct(w0, blk_scale, self.group_size)
            print(f"[MR-GPTQ]   ↳ Hessian not PD → direct MXFP4 "
                  f"in {time.time() - _t0:.1f}s", flush=True)
            return wq.to(orig_device, orig_dtype), None, None
        del hessian

        # 7) 分块 + 逐列量化 + 误差补偿（标准 GPTQ）
        block = self.block_size
        for i1 in range(0, columns, block):
            i2 = min(i1 + block, columns)
            w_block = weight[:, i1:i2]
            err_block = torch.zeros_like(w_block)
            hinv_block = hessian_inv[i1:i2, i1:i2]

            for i in range(i2 - i1):
                w = w_block[:, i]
                d = hinv_block[i, i]
                col_idx = perm[i1 + i]                      # 原始列号
                scale_col = blk_scale[:, col_idx // self.group_size]
                w_q = self._quant_col(w, scale_col)
                err = (w - w_q) / d
                w_block[:, i:] -= err.unsqueeze(1).matmul(hinv_block[i, i:].unsqueeze(0))
                err_block[:, i] = err

            weight[:, i2:] -= err_block.matmul(hessian_inv[i1:i2, i2:])

        # 8) 恢复列顺序；此时 weight 每列已是量化值 → 即 fake-quant 权重
        weight = weight[:, invperm].to(orig_device, orig_dtype)
        print(f"[MR-GPTQ]   ↳ done in {time.time() - _t0:.1f}s", flush=True)
        # mxfp4 无独立 scale/offset（含在共享指数里）
        return weight, None, None

    @torch.no_grad()
    def fake_quant_forward(self, inputs):
        """分块入口。父类状态机在补偿完成的**那一次前向里**就改走本函数出结果喂下游
        （`gptq_module.forward`：cur_batch>batch_num → return fake_quant_forward），
        所以校准期这里拿到的是整个校准 batch，旋转与激活量化的 fp32 中间量同样会 OOM。
        评测期输入是 [1, seq, cin]，dim0=1 不触发分块，行为不变。
        输出写进预分配缓冲区而非 torch.cat——cat 会在拼接瞬间双份占用（up_proj 的
        out=12288 时整块输出就有 6.3GB）。
        """
        if _CALIB_CHUNK and inputs.dim() == 3 and inputs.shape[0] > _CALIB_CHUNK:
            out = None
            off = 0
            for chunk in torch.split(inputs, _CALIB_CHUNK, dim=0):
                r = self._fake_quant_forward_one(chunk)
                if out is None:
                    out = torch.empty((inputs.shape[0], *r.shape[1:]),
                                      dtype=r.dtype, device=r.device)
                out[off: off + r.shape[0]] = r
                off += r.shape[0]
            return out
        return self._fake_quant_forward_one(inputs)

    @torch.no_grad()
    def _fake_quant_forward_one(self, inputs):
        # weight 已是（旋转+）量化后的权重；Hadamard 开时输入需同样旋转，
        # y = (x·R)·(W·R)ᵀ = x·R·Rᵀ·Wᵀ = x·Wᵀ——**只需要 R 正交**（Sylvester 恰好还对称，
        # 但全宽旋转下 down_proj 用的 Paley⊗Sylvester 不对称，不影响正确性）。
        # weight-only 下 x·R 精确无损。
        if self._do_hadamard():
            inputs = self._rotate_last_dim(inputs)
        # W4A4：旋转后把激活也量化到 MXFP4（旋转已压离群值 → 激活可 4-bit）。
        # 注意：只在此 fake-quant 路径量化激活；校准前向走父类 F.linear、不量化激活。
        if self._do_act_quant():
            inputs = self._quant_act_mxfp4(
                inputs, self.group_size, self.increments.get("act_scale_fit", False),
                self.increments.get("act_clip", False),
                self.increments.get("act_fit_rowwise", False))
        return F.linear(inputs, self.weight.to(inputs.dtype), self.bias)


# ---- 融合存储的 MoE 路由专家量化（amct 覆盖不到的大头） --------------------
@torch.no_grad()
def quantize_fused_experts(model, scale_fitting=True, min_experts_numel=1_000_000):
    """对融合存储的 MoE 路由专家做 MXFP4 fake-quant。

    背景：Qwen3.5-MoE 的路由专家不是一堆 nn.Linear，而是 `Qwen3_5MoeExperts` 模块里的
    3D 权重张量 `gate_up_proj`/`down_proj`（形状 [E, out, in]，占 35B 的大头）。amct 只
    量化 nn.Linear → 碰不到它们 → 不处理则「显存降低≥50%」必然不达标。

    做法：沿最后一维(in，收缩维)每 32 元素一组做标准 MXFP4（reshape [E*out, in] 后
    复用逐块量化）。weight-only RTN + 可选 scale fitting；**不做 Hadamard、不做 GPTQ 补偿**。

    不做 Hadamard 的原因：旋转要求推理时对专家输入施加 `x·R`，而本函数走**纯权重改写**
    通路（只重写 `W.data`，不接管 forward），没有挂载输入旋转的位置；要做需自行接管
    `Qwen3_5MoeExperts.forward`。（router 读的是**未旋转**的 hidden state，与旋转无耦合，
    不存在"需与 router 同步"的障碍。）scale fitting 对 RTN 本就是加分项（论文 Table 11）。

    ⚠️口径：本函数覆盖的 ~32B 参数拿到的是 **MXFP4 数据类型**，不是完整 MR-GPTQ 算法。
    报告中的「全参数量化占比 97%」为数据类型覆盖率；完整 MR-GPTQ 只作用于 250 个
    nn.Linear（1.41B，占全参数 4.1%）。

    ⚠️显存：device_map 会把卡塞得很满(card 0 常仅剩 ~1GB)，故计算全程放 **CPU**
    （fp32 临时量 2GB+ 放卡上必 OOM）；回写时先把 CPU 结果赋给 W.data 释放卡上旧张量、
    empty_cache、再挪回卡，保证设备峰值不超基线。

    返回 (专家模块数, 已量化参数量)。对无此结构的模型(如 8B dense)是 no-op。
    """
    def _empty_cache():
        try:
            import torch_npu
            torch_npu.npu.empty_cache()
        except Exception:
            pass

    n_mod = 0
    quant_param = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != "Qwen3_5MoeExperts":
            continue
        n_mod += 1
        done = []                      # 本模块内**实际**完成量化的参数名
        for pname in ("gate_up_proj", "down_proj"):
            W = getattr(mod, pname, None)
            if W is None or W.numel() < min_experts_numel:
                continue
            dev, dt = W.device, W.dtype
            E, O, I = W.shape
            w2 = W.data.reshape(E * O, I).to("cpu").float()  # 先 fp16→CPU 再转 fp32（避免卡上 fp32 临时量）
            if scale_fitting:
                blk = MRGPTQuant._fit_block_scale(w2, 32)          # 附录H 网格重映射
            else:
                blk = torch.pow(2.0, torch.nan_to_num(cal_shared_exponent(w2)))
            wq = MRGPTQuant._quant_direct(w2, blk, 32).reshape(E, O, I).to(dt)  # CPU fp16 结果
            del w2, blk
            W.data = wq                # 先赋 CPU：释放卡上旧权重存储
            _empty_cache()             # 回收，给挪回腾地方
            W.data = wq.to(dev)        # 再挪回原卡（此时峰值不超基线）
            del wq
            quant_param += W.numel()
            done.append(pname)
        # 显存口径的唯一凭据：只有真正走完上面这段的参数才记账。
        # report_memory_full 读此标记，不再按模块类型名无条件计入
        # （否则 min_experts_numel 跳过的、或本函数未被调用时，占比会虚高）。
        mod._mrgptq_fused_quantized = tuple(done)
        _empty_cache()
        print(f"[fused-experts] {name}  (scale_fit={scale_fitting}) 量化完成 "
              f"[{', '.join(done) if done else '无'}]", flush=True)
    print(f"[fused-experts] 共 {n_mod} 个 Experts 模块，量化 {quant_param / 1e9:.2f}B 参数",
          flush=True)
    return n_mod, quant_param


def report_act_coverage(model, skip_layers=("lm_head",)):
    """统计**真 4-bit 激活层占比**——W4A4 的验收口径（按层数）。

    ⚠️ **两个分母都打，别只看宽的那个**：
      · 分母A = 已量化模块数（MRGPTQuant 实例）——回答「能量化的层里有多少跑了真 4-bit」；
      · 分母B = 全部 nn.Linear（含 amct 量化不了的，跳 skip_layers）——**任务书原文口径**
        （「量化层（torch.nn.Linear）占比不低于 70%」），验收以此为准。
    8B 上两者几乎重合（252 vs 253），差别可忽略；**35B 上是 250 vs 350，相差 100 层，
    足以决定达标与否**（保护 down_proj/o_proj 时 200/250=80% 但 200/350=57.1%）。

    分子掉出的两类：
      ① `_ACT_PROTECT` 命中（混合精度，主动保高精度激活）；
      ② in_features 不被 hadamard k 整除、拿不到旋转（W4A4 需旋转压离群值，见 _do_act_quant）。
    未量化的 nn.Linear（in/out 不整除 32，做不了 block-32 MXFP4）只进分母 B。
    """
    on, off_prot, off_rot = [], [], []
    n_quant = n_unquant = 0
    for name, mod in model.named_modules():
        if isinstance(mod, MRGPTQuant):
            n_quant += 1
            if not mod.increments.get("act_quant"):
                continue
            if any(p in mod.layer_name for p in _ACT_PROTECT):
                off_prot.append(mod.layer_name)
            elif mod.increments["hadamard"] and not mod._do_hadamard():
                off_rot.append(mod.layer_name)
            else:
                on.append(mod.layer_name)
        elif isinstance(mod, torch.nn.Linear) and not any(s in name for s in skip_layers):
            n_unquant += 1  # amct 量化不了的层（维度不整除 32），只进分母 B
    total_all = n_quant + n_unquant
    pa = 100.0 * len(on) / max(n_quant, 1)
    pb = 100.0 * len(on) / max(total_all, 1)
    print(f"[act-coverage] 真 4-bit 激活层 / 已量化模块 = {len(on)}/{n_quant} = {pa:.1f}%")
    print(f"[act-coverage] 真 4-bit 激活层 / 全部 nn.Linear = {len(on)}/{total_all} = {pb:.1f}%"
          f"  ← **任务书口径，要求≥70%** [{'PASS' if pb >= 70 else 'FAIL'}]")
    if n_unquant:
        print(f"[act-coverage]   ↳ 两分母差 {n_unquant} 层：amct 量化不了"
              f"（in/out 不整除 32，无法 block-32 MXFP4），永远进不了分子")
    if off_prot:
        print(f"[act-coverage]   ↳ 混合精度保护跳过 {len(off_prot)} 层（--act_protect {_ACT_PROTECT}）")
    if off_rot:
        print(f"[act-coverage]   ↳ 无法旋转而跳过 {len(off_rot)} 层（in_features 不整除 k）："
              f"{sorted(set(n.split('.')[-1] for n in off_rot))}")
    return len(on), n_quant, total_all, pb


# ---- 运行时注册 'mr_gptq'（零改 amct 源码） --------------------------------
def register():
    import amct_pytorch.algorithms as _algos
    import amct_pytorch.common.utils.vars as _vars
    from amct_pytorch.classic.deploy_op.weight_npu_quant_module import NpuWeightQuantizedLinear

    # 1) 注册算子 + 加入内置算法白名单
    _algos.AlgorithmRegistry.register("mr_gptq", "Linear", MRGPTQuant, NpuWeightQuantizedLinear)
    if "mr_gptq" not in _algos.BUILT_IN_ALGORITHM:
        _algos.BUILT_IN_ALGORITHM.append("mr_gptq")

    # 2) 加入 (算法, 量化类型组合) 支持表：weight-only MXFP4
    comb = "NOT_QUANTIZE mxfp4_e2m1"
    supported = _vars.ALGORITHM_SUPPORTED_QUANT_TYPE_COMB.get(comb)
    if supported is not None and "mr_gptq" not in supported:
        supported.append("mr_gptq")


register()
