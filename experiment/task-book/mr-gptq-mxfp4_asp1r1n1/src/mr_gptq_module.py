# -*- coding: UTF-8 -*-
"""MR-GPTQ（Micro-Rotated GPTQ, arXiv:2509.23202）MXFP4 实现。

子类化 amct 的 GPTQuant，override 权重优化，做三件事：
  1. **标准 block-32 MXFP4 量化**（沿 in-features、每 32 元素共享 E8M0 指数），
     取代 stock 的 per-column `.repeat(1,2)` 非标准缩放；
  2. **shared exponent 预计算一次 + CPU 上做补偿**——消除 stock 每列重算 20+ 小算子、
     百万次 NPU kernel 启动的开销（2h → 分钟级）；
  3. MR-GPTQ 三增量（按开关渐进加入）：
       - block-wise Hadamard 微旋转
       - MXFP scale fitting（附录 H）
       - 静态激活重排
     本文件先落地 ①②（= 干净的 GPTQ-mxfp4 baseline），增量在 _INCREMENTS 开关下逐个加。

零改 amct 源码：import 本模块即运行时注册 'mr_gptq'。
"""
import torch
import torch.nn.functional as F

from amct_pytorch.classic.quantize_op.gptq_module import GPTQuant
from amct_pytorch.common.utils.quant_util import cal_shared_exponent
from amct_pytorch.common.utils.data_utils import float_to_fp4e2m1
from amct_pytorch.common.utils.vars import MXFP4_E2M1


# MR-GPTQ 增量开关（默认全关 = GPTQ-mxfp4 baseline；逐个打开做消融）
_INCREMENTS = {
    "hadamard": False,        # block-wise Hadamard 微旋转
    "scale_fitting": False,   # MXFP scale fitting（附录 H）
    "act_reorder": False,     # 静态激活重排
}

_HADAMARD_K = 128  # block-wise Hadamard 尺寸（论文示 MXFP4 上 128 优于 32）


def _sylvester_hadamard(k):
    """归一化 Sylvester-Hadamard H_k/√k（k 须为 2 的幂），正交且对称。"""
    H = torch.ones(1, 1)
    while H.shape[0] < k:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / (k ** 0.5)


class MRGPTQuant(GPTQuant):
    """MR-GPTQ 量化算子。当前实现 = 高效标准 MXFP4 + GPTQ 误差补偿 + 可选增量。"""

    def __init__(self, ori_module, layer_name, quant_config):
        super().__init__(ori_module, layer_name, quant_config)
        self.increments = dict(_INCREMENTS)
        if self.wts_type != MXFP4_E2M1:
            raise ValueError(f"MR-GPTQ 目前仅支持 mxfp4_e2m1，收到 {self.wts_type}")
        self.hadamard_k = _HADAMARD_K
        self._H = _sylvester_hadamard(self.hadamard_k)  # [k,k] CPU float32

    # ---- block-wise Hadamard 旋转（增量①） --------------------------------
    def _do_hadamard(self):
        # in_features 需被 k 整除；否则该层跳过旋转（Qwen3 各层均可整除）
        return self.increments["hadamard"] and self.weight.shape[1] % self.hadamard_k == 0

    def _rotate_last_dim(self, x):
        """沿最后一维（in）做块对角 Hadamard：reshape [...,n_blk,k] @ H_k。"""
        k = self.hadamard_k
        H = self._H.to(device=x.device, dtype=x.dtype)
        shape = x.shape
        x = x.reshape(*shape[:-1], shape[-1] // k, k).matmul(H)
        return x.reshape(shape)

    def update_hessian(self, input_data):
        # 旋转后累积 Hessian，使 H_r = RᵀHR 与旋转权重一致（转输入比转 Hessian 便宜）
        if self._do_hadamard():
            input_data = self._rotate_last_dim(input_data)
        super().update_hessian(input_data)

    # ---- MXFP scale fitting（附录 H）：每 block 搜 MSE 最优共享指数 ---------
    @staticmethod
    def _fit_shared_exponent(weight, block_size=32, search=(-2, -1, 0, 1)):
        """标准 cal_shared_exponent 给基准指数，再在 base+search 候选里
        逐 block 选重构 MSE 最小的指数（仍是 E8M0 幂次，只是更贴数据分布）。
        返回 [out, n_groups]，与 cal_shared_exponent 同形，可直接按列索引。
        """
        base = torch.nan_to_num(cal_shared_exponent(weight, block_size))  # [out, n_groups]
        out, in_features = weight.shape
        n_groups = base.shape[1]
        pad = n_groups * block_size - in_features
        w = F.pad(weight, (0, pad)) if pad else weight
        w = w.reshape(out, n_groups, block_size)                          # [out, n_groups, 32]

        best_se = base.clone()
        best_err = torch.full_like(base, float("inf"))
        for d in search:
            se = base + d
            scale = torch.pow(2.0, se).unsqueeze(-1)                      # [out, n_groups, 1]
            q = float_to_fp4e2m1(w / scale) * scale
            err = ((w - q) ** 2).sum(dim=-1)                             # [out, n_groups]
            better = err < best_err
            best_se = torch.where(better, se, best_se)
            best_err = torch.where(better, err, best_err)
        return best_se

    # ---- 标准 block-32 MXFP4 单列量化 -------------------------------------
    @staticmethod
    def _quant_col_mxfp4(w, se_col):
        """对单列 w[out] 用其组共享指数 se_col[out] 做 MXFP4 fake-quant。

        w_scaled = w · 2^(-se) → round 到 E2M1 网格 → · 2^(se)。
        """
        scale = torch.pow(2.0, se_col)
        q = float_to_fp4e2m1(w / scale)
        return q * scale

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

        # 全程 CPU float32：小算子无 NPU 启动开销，Cholesky 亦在 CPU（~0.7s）
        weight = self.weight.detach().to("cpu", torch.float32)
        if self._do_hadamard():
            weight = self._rotate_last_dim(weight)  # 旋转权重（Hessian 已在 update 时旋转）
        hessian = self.hessian.to("cpu", torch.float32)

        # 1) 死列（Hessian 对角为 0）处理
        diag = torch.diag(hessian)
        dead = diag == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        # 2) 预计算 block-32 shared exponent（沿 in-features），静态、只算一次
        #    形状 [out, ceil(in/32)]；后续每列按 原始列号//32 取用
        #    增量①：scale fitting → 逐 block 搜 MSE 最优指数
        if self.increments["scale_fitting"]:
            shared_exp = self._fit_shared_exponent(weight)
        else:
            shared_exp = torch.nan_to_num(cal_shared_exponent(weight))

        # 3) 激活重排：按 Hessian 对角（激活能量）排序处理列。
        #    stock GPTQ 亦如此；MR-GPTQ 的“静态激活重排”增量将在此细化。
        perm = torch.argsort(diag, descending=True)
        invperm = torch.argsort(perm)
        weight = weight[:, perm]
        hessian = hessian[perm][:, perm]

        # 4) Hessian 逆的上三角 Cholesky（override 版，全程 CPU）
        hessian_inv = self.cal_hessian_inverse(hessian, columns)
        del hessian

        # 5) 分块 + 逐列量化 + 误差补偿（标准 GPTQ）
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
                se_col = shared_exp[:, col_idx // self.group_size]
                w_q = self._quant_col_mxfp4(w, se_col)
                err = (w - w_q) / d
                w_block[:, i:] -= err.unsqueeze(1).matmul(hinv_block[i, i:].unsqueeze(0))
                err_block[:, i] = err

            weight[:, i2:] -= err_block.matmul(hessian_inv[i1:i2, i2:])

        # 6) 恢复列顺序；此时 weight 每列已是量化值 → 即 fake-quant 权重
        weight = weight[:, invperm].to(orig_device, orig_dtype)
        # mxfp4 无独立 scale/offset（含在共享指数里）
        return weight, None, None

    @torch.no_grad()
    def fake_quant_forward(self, inputs):
        # weight 已是（旋转+）量化后的权重；Hadamard 开时输入需同样旋转，
        # y = (x·R)·(W·R)ᵀ = x·Wᵀ（R 正交对称），weight-only 下 x·R 精确无损。
        if self._do_hadamard():
            inputs = self._rotate_last_dim(inputs)
        return F.linear(inputs, self.weight.to(inputs.dtype), self.bias)


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
