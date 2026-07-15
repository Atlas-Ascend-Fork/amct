# 低比特量化 / MR-GPTQ 技术笔记（"能讲清楚"版）

> **这份文档的用法**：不是拿来读爽的，是拿来**主动回忆**的。
> 每节结构固定：**一句话答案 → 关键公式/洞察 → 代码在哪 → 🎯自测**。
> 读法：先盖住答案自己讲一遍，讲不出来再看。第 7 节是面试模拟题，**必须脱稿答**。
> 标 ⚠️ 的是反直觉、最容易被追问穿的点；标 ✍️ 的是你得能**自己在纸上推**的（对应第 8 节线代清单）。
>
> 配套：算法方案见 [design.md](design.md)，实验流水见 [experiment-log.md](experiment-log.md)，代码在 [../src/](../src/)。

---

## 0. 电梯陈述（背下来）

> 在华为昇腾的 amct 量化框架上，为 **MXFP4（4-bit 浮点）权重量化**复现并实现了 SOTA 算法 **MR-GPTQ**。
> 把 Qwen3-8B 的 PPL 劣化从纯 GPTQ 的 **+0.42（不达标）压到 +0.27（达标）**，量化层覆盖 99.6%、权重显存降 67%；
> 同时把量化耗时从框架自带实现的 **2–3 小时降到几分钟（~15×）**。核心贡献是三个：
> ① 想清楚了"为什么在 MXFP4 上 GPTQ 几乎等于 RTN"，② 用 Hadamard 微旋转 + scale fitting 两个增量精准打这个痛点，
> ③ 把逐列量化从 NPU 搬到 CPU 消除 kernel 启动开销。

一句话钩子（面试开场）："**这个项目的价值不在于跑通了一个算法，而在于我搞清楚了 4-bit 浮点量化的误差到底卡在哪、以及为什么通用的 GPTQ 在这里失效。**"

---

## 1. 底座：MXFP4 是什么，为什么难

**一句话**：MXFP4 = 每 32 个权重共享一个"只能是 2 的幂"的缩放因子（E8M0），每个权重本身用 4-bit 浮点（E2M1）表示。

**E2M1 元素网格**（1 符号 + 2 指数 + 1 尾数，共 4-bit，16 个值里去掉重复，实际 8 个绝对值）：
```
{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
```
注意它**不均匀**：小值密（0.5 间隔）、大值疏（4→6 跳 2）。所以塞进这个网格的数，最好分布集中在小值区。

**E8M0 共享指数**：一组 32 个权重共享一个缩放 `scale`，量化时 `w/scale` 落到 E2M1 网格，反量化 `w_q * scale`。E8M0 是 8-bit **纯指数**，意味着 `scale` **只能取 2^k**（power-of-two, PoT），k∈[-127,128]。

⚠️ **头号洞察：E8M0 的 PoT 约束是精度杀手。**
理想 scale 应该让 `block_max/scale` 正好顶到 6（用满 E2M1 范围）。但 scale 只能是 2 的幂 → 实际 scale 往往比理想值大（向上取到最近的 2^k）→ 整块权重被压小、挤到 E2M1 的小值密集区还好，但更常见的是浪费了 [4,6] 这段动态范围 → 量化误差偏大。**MXFP4 的误差主要来自 scale 太粗，而不是单个元素的 round。** 这句话是整个 MR-GPTQ 的立论基础。

**代码**：`cal_shared_exponent`（算 2^k 的 k）+ `float_to_fp4e2m1`（把 `w/scale` round 到那 8 个值，纯布尔掩码，见 [mr_gptq_module.py](../src/mr_gptq_module.py) 的 `_quant_col`/`_quant_direct`）。

🎯 **自测**：
- E2M1 能表示的最大值是多少？为什么 scale fitting 里用 `log2(block_max) - log2(6)` 而不是 `-log2(4)`？
- 为什么说"MXFP4 的误差主要来自 scale 而非元素 round"？

---

## 2. GPTQ 在干嘛：逐列量化 + 二阶误差补偿

**一句话**：GPTQ 一列一列地量化权重，每量化一列就产生误差，它用 Hessian 的逆把这个误差**摊派**到还没量化的列上，让"量化后整层输出的变化"最小。

✍️ **目标函数**（layer-wise reconstruction）：最小化
```
|| W·X − Ŵ·X ||²_F        （W 原权重，Ŵ 量化权重，X 是校准输入）
```

✍️ **Hessian 的来历**：把上式对 Ŵ 的一行求二阶导 → `H = 2·X·Xᵀ`。
代码里 `update_hessian` 累积的就是它（带 1/nsamples 归一化 + √(2/n) 因子）：`H = (2/n)·Σ xxᵀ`。
直觉：`H` 描述"哪些输入通道能量大 / 相关"——能量大的通道，权重量化错了对输出影响大，要优先照顾。

**补偿公式（OBQ/OBS）**：量化第 i 列得到 `w_q`，误差 `err = (w_i − w_q) / [H⁻¹]_ii`，然后把
`−err · H⁻¹[i, i:]` 加到后面所有未量化列上。这样在**校准分布上**，这一列的量化误差被后续列"吸收"掉了。

**Cholesky 的作用**：直接反复取 `H⁻¹` 的列做上面的更新数值不稳。GPTQ 先对 `H⁻¹` 做**上三角 Cholesky 分解**，一次性把"误差传播系数"稳定地预计算进一个三角矩阵，之后逐列/分块取用。

**代码**：[mr_gptq_module.py](../src/mr_gptq_module.py) `get_opt_weight_and_quant_factor`——第 7 步的双重循环就是 OBQ 更新；`cal_hessian_inverse` 就是 damping + Cholesky。

⚠️ 注意：GPTQ **只在校准分布 X 上**最小化输出误差。所以校准数据的代表性很重要（后面 MoE 冷 expert 的坑就是从这来的）。

🎯 **自测**：
- 为什么 Hessian 是 `2XXᵀ` 而不是 `XᵀX`？维度是多少（in×in 还是 out×out）？
- 补偿为什么只摊给"未量化"的列，不摊给已量化的？
- 如果 `H` 某个对角元是 0 说明什么？代码怎么处理（`dead` 那几行）？

---

## 3. ⚠️ 最反直觉的一点：MXFP4 上 GPTQ ≈ RTN

**现象**（实验 E1 vs E1'）：我们的 GPTQ-mxfp4 = 10.1332，纯 RTN（论文 Table 11）几乎一样，delta +0.42 **不达标**。GPTQ 这套二阶补偿在这儿几乎白干。

**为什么**：
1. MXFP4 的 block=32 已经很细，**局部 scale 自适应**做得不错，逐元素 round 误差本来就不大 → GPTQ 能补偿的"元素误差"没多少油水。
2. 剩下的主要误差来自 **E8M0 的 PoT scale 约束**（见 §1）。GPTQ 调的是**权重值**，而 scale 被死死限制成 2 的幂——**GPTQ 再怎么摊派误差，也逃不出这个 PoT 网格。**

✅ **推论（MR-GPTQ 的整个 motivation）**：要提精度，不能靠"更狠地补偿"，得去**动 scale（让它不再是纯 2 的幂）和 动权重分布（让它更适配 E2M1 网格）**。这就是下面两个增量。

🎯 **自测**：如果面试官问"你都上 GPTQ 了怎么还不达标"，你怎么用一句话解释？（答：GPTQ 治的是元素 round 误差，但 MXFP4 的误差瓶颈是 scale 的 PoT 约束，GPTQ 够不着。）

---

## 4. MR-GPTQ 三增量：各自打哪个痛点

### 4.1 Hadamard 微旋转 —— 治"权重分布"

✍️ **数学**：Linear 算的是 `y = x·Wᵀ`。插一个**正交对称**矩阵 R（R = Rᵀ = R⁻¹，所以 R·Rᵀ = I）：
```
y = (x·R)·(W·R)ᵀ = x·R·Rᵀ·Wᵀ = x·Wᵀ      ← 输出精确不变
```
于是：**权重存旋转后的 `W·R`，推理时输入也旋转 `x·R`**，两边抵消。weight-only 下 `x·R` 是精确无损的线性变换。

**物理直觉**：Hadamard 是"全是 ±1/√k 的正交混合矩阵"，把一个 block 内的通道**均匀搅拌**。搅拌后权重值更接近高斯/均匀、**离群值被摊平** → 塞进 E2M1 网格（小值密）时 clipping/round 损失更小，PoT scale 也更贴合。**本质是在量化前把分布"整形"成量化器喜欢的样子。**

✍️ **Hessian 也要转**：旋转空间里 `H_r = Rᵀ·H·R`。代码里没有直接转 H（贵），而是**在累积 Hessian 前先转输入**（`update_hessian` 里 `_rotate_last_dim(input)`）——数学上等价，更便宜。

⚠️ **为什么 k=128 > 32**（论文结论）：量化 block 是 32，但 Hadamard 取 128 = **跨 4 个量化 block 搅拌**，离群值打散得更彻底。k 是旋转粒度，不等于量化粒度。

**代码**：`_sylvester_hadamard`、`_rotate_last_dim`、`_do_hadamard`、`update_hessian`、`fake_quant_forward`（推理时转输入）。

### 4.2 MXFP scale fitting —— 治"scale 太粗"（论文附录 H 式2/3）

**痛点**：E8M0 有 256 个码，但它们摊在 `[-127,128]` 整个指数范围，**实际权重的 scale 只用到其中很窄一段**，绝大多数码浪费了。

✍️ **做法（网格重映射，不是 MSE 搜索！）**：把 256 个码**重新映射到数据实际的 log-scale 范围** `[Lmin, Lmax]`：
```
L      = log2(block_max) − log2(6)          # 每块理想 log-scale（block_max 映到 E2M1 上限 6）
Lmin,Lmax = L.min(), L.max()                # across all blocks
q      = clamp(round(255·(L−Lmin)/(Lmax−Lmin)), 0, 255)   # 256 级细网格
L_fit  = Lmin + q·(Lmax−Lmin)/255
scale  = 2^L_fit                            # 注意：步长 α=(Lmax−Lmin)/255 < 1
```
关键：`scale = 2^(α·q + β)`，**α < 1 → scale 不再是整数次幂（sub-power-of-two）** → scale 分辨率变细 → §1 说的 scale 量化误差变小。

⚠️ **最大的坑（也是最能讲的点）**：scale fitting **单独用会伤害纯 GPTQ**（论文 Table 11：Qwen3-8B 94.1→92.3）。因为权重分布没整形时，细 scale 会去"迁就离群值"反而更糟。**必须叠在 Hadamard 之上才生效**（94.1→98.5 是 MR-GPTQ 全量的数）。→ 这说明三增量不是独立叠加，是**有依赖关系的组合**。
另一个我踩过的坑：offset 用 `log2(6)` 而不是 `log2(4)`——用 4 的话 block_max 只映到 4，浪费了 E2M1 的 [4,6] 段，反而变差（实测 10.094 vs 10.050）。

**代码**：`_fit_block_scale`。

### 4.3 静态激活重排（act_reorder）—— 次要增量

按 Hessian 对角（激活能量）排序处理列，能量大的先量化、先被照顾。stock GPTQ 里 `perm = argsort(diag)` 已经做了基础版；MR-GPTQ 的"静态"版把这个顺序固化。当前实现里是占位/基础态，收益最小。

🎯 **自测**：
- Hadamard 为什么能保证输出不变？R 需要满足什么性质？（正交 + 对称）
- scale fitting 为什么单独用会变差、和 Hadamard 一起才好？
- 如果只能保留一个增量，你留哪个？为什么？（Hadamard——它是 MXFP4 头号杠杆，纯 GPTQ 过不了门槛、加它就过了）

---

## 5. 工程决策（AI infra 能力的体现点）

**① fake-quant 是纯 torch，不需要 NPU 原生算子。**
MXFP4 的量化 = `cal_shared_exponent`(log2/floor) + `float_to_fp4e2m1`(布尔掩码) + 乘 scale，全是逐元素张量运算。搞清这点后，整个算法在**任何设备（甚至 CPU）都能跑**，也不依赖 `amct_ops`（对比 hifloat8 需要 `amct_ops.hifloat8_cast`）。精度由确定性数学决定，与 NPU 型号无关——所以 910B 上的 PPL 和目标 A5 等效。

**② CPU 补偿带来 ~15× 提速——关键是想清楚慢在哪。**
框架自带 GPTQ 跑 MXFP4 要 2–3 小时。逐个排查：Cholesky 只要 0.7s、逐列循环 1.3s——都不是瓶颈。真凶是**每列都在 NPU 上做一次 MXFP4 量化**（`quant_dequant_tensor` 带 `.repeat(1,2)`），几百万次微算子 → **NPU kernel 启动开销**堆死。
解法：把整个补偿搬到 **CPU float32**（小算子无启动开销）+ **每块 scale 只预计算一次**（不再每列重算）→ 分钟级。
⚠️ 讲这个点时强调**方法论**：性能问题先测量定位、别猜；瓶颈常常不在"看起来重"的那步（Cholesky）而在"看起来轻但调用次数爆炸"的那步（kernel launch）。

**③ 运行时注册，零改 amct 源码。**
`MRGPTQuant(GPTQuant)` 子类化 + `register()` 里 `AlgorithmRegistry.register` + 追加到 `BUILT_IN_ALGORITHM` 和 `ALGORITHM_SUPPORTED_QUANT_TYPE_COMB` 白名单。`import` 模块即注册。好处：交付干净、不污染框架、升级不冲突。

🎯 **自测**：
- 为什么把逐列量化从 NPU 挪到 CPU 反而快？（不是 CPU 算得快，是省了百万次 kernel 启动）
- 排查性能瓶颈时，你怎么确定不是 Cholesky 的锅？（分段计时）

---

## 6. MoE 特有的坑（35B，本次新增）

**背景**：MoE 每个 token 只路由到 top-k 个 expert，64 条校准样本 → 很多 expert 拿到的 token 很少甚至为 0。GPTQ 依赖校准激活算 Hessian，这直接出问题。

**① 冷 expert（一个 token 都没分到）**：它的量化模块 `forward` 从不触发 → `cur_batch` 停在 0 → 补偿逻辑 `get_opt` 从不执行 → **权重保持 fp16 没被量化**。不崩，但**真实覆盖率 < 量化前统计的理论覆盖率**。
→ 解法：跑完用 `report_actual_coverage`（按 `cur_batch≥2` 判定）复核**真实**覆盖率，把冷 expert 揪出来；不够就加 `n_calib` 点亮更多 expert。

**② 稀疏 expert（只被极少 token 命中）**⚠️ 真正的雷：Hessian 秩亏 → 大量对角为 0（dead 列）。原代码 `weight[:, dead] = 0` 是 stock GPTQ 的"死列清零"——对 dense 层只有个位数死列无所谓，但对稀疏 expert 会**把整块权重清零，等于毁掉这个 expert**（它在真实分布上是会被用的！）。还可能 Cholesky 非正定报错。
→ 解法：`dead > 50%` 或 Hessian 非正定时，**退化为直接 MXFP4 量化（RTN，不补偿不清零）**——`_quant_direct`。RTN 虽不如 GPTQ，但远好过清零。

**③ 为什么 batch_num=1 单次大 forward 是对的**⚠️：量化触发靠 `cur_batch == batch_num`。`batch_num=1` 时，**任何被命中过一次的 expert 都会触发量化**。若为省显存改成 K 个 micro-batch（batch_num=K），则只有在**全部 K 次 forward 里都被路由到**的 expert 才触发——绝大多数 expert 会卡在 fp16。**所以单次大 forward 恰恰是 MoE 的正确设计，不能拆。** 显存/覆盖率的平衡改用调 `n_calib` 解决。

**④ 多卡（device_map）**：
- 父类 `forward` 开头 `inputs = inputs.to(self.weight.device)`——**每个量化 Linear 自己把输入搬到自己权重所在卡**，跨卡路由天然成立，量化模块不用额外改。
- 非量化模块（norm 等）由 accelerate 的 hook 搬设备。
- ⚠️ **CPU/disk offload 会崩**：amct 换层后不会加载被 offload 的权重。所以 `get_model` 强制校验 `hf_device_map` 里没有 cpu/disk，显存不够就报错提示加卡。
- PPL 计算时 lm_head 输出可能落在另一张卡 → labels 要 `.to(logits.device)` 对齐。

**⑤ skip_layers 是子串匹配的陷阱**⚠️：amct `check_skip_layer` 是 `if skip_layer in layer_name`（子串）。要跳过 MoE router 又不误伤 expert：
- router 名 `...mlp.gate`，expert 名 `...mlp.experts.N.gate_proj`。
- 子串 `"mlp.gate"` 命中 router，但**不**命中 `...mlp.experts.N.gate_proj`（那段是 `mlp.experts`）。✅
- 但 `"gate"` 或 `".gate"` 会命中所有 `gate_proj` ❌——差一点就把 1/3 的 expert 跳掉了。

**代码**：[mr_gptq_module.py](../src/mr_gptq_module.py) `_quant_direct` + `get_opt` 的三层兜底；[eval_utils.py](../src/eval_utils.py) `get_model`/`input_device`/`report_actual_coverage`；[run_eval.py](../src/run_eval.py) `--device_map`/`--skip_router`。

🎯 **自测**：
- 冷 expert 和稀疏 expert 的处理为什么不一样？（冷的根本没触发、天然留 fp16；稀疏的触发了但会毁权重，要主动兜底）
- 为什么不能用 micro-batch 校准省显存？
- `"mlp.gate"` 能跳 router 不误伤 expert，靠的是什么？

---

## 7. 面试模拟题（脱稿答，答不出就回去看对应节）

1. 用 30 秒讲清这个项目做了什么、你的贡献。（§0）
2. MXFP4 的误差主要来自哪？为什么？（§1）
3. GPTQ 的数学目标是什么？Hessian 从哪来？（§2）
4. 你上了 GPTQ 为什么还不达标？（§3，⚠️必答）
5. Hadamard 旋转为什么不改变输出？物理上它解决什么问题？（§4.1）
6. scale fitting 为什么单独用反而变差？（§4.2，⚠️）
7. 你怎么把量化从 2 小时优化到几分钟？瓶颈怎么定位的？（§5，⚠️infra 岗必问）
8. MoE 量化你遇到什么坑？"稀疏 expert 清零"这个 bug 怎么发现和修的？（§6，⚠️）
9. 多卡跨卡张量你怎么保证不出 device mismatch？（§6④）
10. 如果 delta 还是超 0.4，你下一步会调什么？（加 Hadamard→加 n_calib→跳 router→提校准样本，按投产比排序）

---

## 8. ✍️ 你得能自己在纸上推的（线代补课清单）

复习线代时重点对上这几个——它们不是"用过的库"，是这个项目的**数学地基**，面试官一追就见真章：

- **正交矩阵**：R 正交 ⇔ RᵀR = I ⇔ 保内积/保范数。Hadamard 为什么正交？归一化的 `H_k/√k` 为什么行/列两两正交。（§4.1）
- **对称 + 正交**：R = Rᵀ 且正交 ⇒ R = R⁻¹（对合）。为什么"旋转权重 + 旋转输入"能抵消。（§4.1）
- **二次型与 Hessian**：`||(W−Ŵ)X||²` 对权重的二阶导为什么是 `2XXᵀ`；`XXᵀ` 为什么半正定；对角元的物理含义。（§2）
- **Cholesky 分解**：正定矩阵 = LLᵀ 的意义；为什么用它解/求逆比直接求逆稳；"非正定 → 分解失败"的判定。（§2、§6②）
- **秩与秩亏**：为什么"某 expert 只见了几个 token" ⇒ `XXᵀ` 秩亏 ⇒ 有零特征值 ⇒ 对角出现 0（dead 列）。（§6②）
- **基变换视角**：Hadamard 旋转 = 换一组正交基看权重；量化误差在不同基下的分布不同——这是"旋转能整形分布"的本质。（§4.1）

> 复习完线代回来，试着**不看这份文档**把第 4、7 节讲一遍。讲得出来，这个项目就真是你的了。
