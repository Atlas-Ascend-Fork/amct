# AMCT HiFloat8 量化体验报告

> Qwen3-0.6B + Wikitext-2 | Ascend910_9362 | CANN 9.0.0 | AMCT 1.1.0

---

## 1.

本次在两张 `Ascend910_9362` 设备、CANN 9.0.0、AMCT 1.1.0 环境中，
完成了 Qwen3-0.6B 的 BF16 基线、HiFloat8 Cast、Quantile、OFMR 量化态模拟和
Wikitext-2 全量 PPL 对比，并在 NPU 上实测 HiFloat8 编解码算子性能。

> **结果：**
>
> - BF16 PPL：`19.1651208`。
> - 最优方案为 Quantile，PPL：`19.2813812`，绝对增量 `+0.1162604`，相对增幅
>   `+0.6066%`。
> - 官方 Cast/Quantile/OFMR 配置并非严格控制变量；补充统一 weight/input 粒度和
>   `skip_layers={'lm_head'}` 后，Quantile 仍然最优。
> - 16M 元素 BF16 实测 encode/decode 吞吐约 `3.78/3.80 GB/s`，roundtrip 有效吞吐约
>   `2.53 GB/s`。
> - AMCT 可通过 `amct_ops` fallback 完成真实 HiFloat8 encode/decode 量化误差模拟，
>   但当前 CANN 9.0.0 原生
>   `aclnnQuantize` 不支持 `DT_HIFLOAT8`，因此 `amct.convert()` 无法生成可运行的原生
>   部署模型。

权重和激活确实经过了 `BF16 -> uint8 HiFloat8 code -> BF16` 的有损 roundtrip，再送入
BF16 Linear，所以 PPL 中包含真实的 HiFloat8 舍入误差；但计算核仍是 BF16 Linear，
这是由于原生 HiFloat8 MatMul 未支持。。

![本次实际验证的 HiFloat8 执行路径](figures/hifloat8_execution_path.png)

---

## 2. 环境说明

### 2.1 硬件与系统

| 项目 | 实测值 |
| --- | --- |
| OS | Ubuntu 20.04.5 LTS，aarch64 |
| CPU | 40 核，单 NUMA 节点 |
| 主机内存 | 229 GiB |
| NPU | 2 个设备，`Ascend910_9362` |
| 单设备 HBM | 约 62.7 GiB |
| 单设备核心 | 20 Cube Core，40 Vector Core |
| 本次实验设备 | `npu:0` |

### 2.2 软件版本

| 组件 | 版本/基线 |
| --- | --- |
| CANN | 9.0.0 |
| CANN 路径 | `/home/developer/Ascend/cann-9.0.0` |
| NPU Driver | 25.5.5 |
| Python | 3.11.4 |
| GCC / CMake | 9.4.0 / 3.20.5 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1.post4 |
| AMCT | amct-pytorch 1.1.0 |
| amct_ops | 1.0.0 |
| Transformers | 4.51.0 |
| datasets | 4.8.4 |

仓库根 `requirements.txt` 固定 Transformers 5.12.1，但本机系统 torchaudio 2.11 与
PyTorch 2.7.1 ABI 不一致，加载 Qwen3 时会失败。模型 `config.json` 以及仓库 Qwen3
样例均使用 4.51.0，因此正式实验采用独立的 4.51.0 环境。

### 2.3 确认环境流程如下

```bash
cd /mnt/workspace
uname -a
cat /etc/os-release
lscpu
free -h
df -h /mnt/workspace

npu-smi info
npu-smi info -m
ls -l /dev/davinci* /dev/davinci_manager

source /home/developer/Ascend/cann-9.0.0/set_env.sh
printf 'ASCEND_HOME_PATH=%s\n' "$ASCEND_HOME_PATH"
cat "$ASCEND_HOME_PATH/version.cfg"
gcc --version
cmake --version

python3 --version
python3 -c "import torch, torch_npu; \
print(torch.__version__, torch_npu.__version__); \
print(torch.npu.is_available(), torch.npu.device_count()); \
print(torch.npu.get_device_name(0)); \
print(torch.npu.get_device_properties(0))"
```

其中，`torch.__version__` 显示 `2.7.1+cpu` 并不代表不能
使用 NPU，Ascend 是通过 `torch_npu` 的 PrivateUse1 后端接入，我用小 NPU benchmark乘法确认
了运行链路。

```bash
python3 -c "import torch, torch_npu; \
x=torch.tensor([1.,-1.],dtype=torch.float16,device='npu:0'); \
print((x*2).cpu())"
```

> 输出为 `tensor([ 2., -2.], dtype=torch.float16)`，说明在引入 AMCT 前，基础 NPU
> runtime 已经可用。

---

## 3. HiFloat8 与 AMCT 核心逻辑

### 3.1 HiFloat8 转换算子

Python 侧的 HiFloat8 编码是 `torch.uint8` Tensor，不是 PyTorch 原生浮点 Tensor：

```text
FP16/BF16 --encode_to_hifloat8--> uint8 HiFloat8 code
FP16/BF16 <--decode_from_hifloat8-- uint8 HiFloat8 code
```

该格式动态分配指数和尾数位：接近 1 的区域保留更多尾数精度，极端数值使用更多指数
位。实现将复杂格式转换放在 Host 侧预计算：

- FP16/BF16 encode LUT：各 `32768 x 1 B`，利用符号对称性只保存 15 bit magnitude。
- FP16/BF16 decode LUT：各 `256 x 2 B`。
- NPU Kernel 只执行位操作、LUT 查询和数据搬运。
- 最大有限值为 32768，最小非零绝对值为 `2^-22`；有限溢出饱和，NaN 归一为单一码。

Atlas A2/A3 不原生支持 HiFloat8 dtype 时，需要编译安装 `amct_ops/hifloat8_cast`。
当前 master 会实际探测 native 能力，失败后在量化阶段回退到 `amct_ops`。

### 3.2  fallback 工作路径

我最初把“存在 `torch_npu.hifloat8`”理解成“系统原生一定支持”。实测 native cast
返回 161002 ，我回到 master 的 `amct_pytorch/quantization/dtypes/hifp_impl.py`
逐行确认。发现是先用 1 个 FP16 零值做最小 roundtrip；只有
roundtrip 真成功才走 native，否则加载 `amct_ops`。核心逻辑：

```python
@torch.no_grad()
def hifloat8_fake_quant(fp_tensor):
    if is_native_hifloat8_cast_available():
        try:
            return _native_hifloat8_fake_quant(fp_tensor)
        except (RuntimeError, OSError):
            pass

    try:
        ops = _load_amct_ops_cast()
    except (RuntimeError, OSError):
        ops = None
    if ops is not None:
        try:
            return _amct_ops_hifloat8_fake_quant(fp_tensor, *ops)
        except (RuntimeError, OSError):
            pass
    raise RuntimeError("native HiFloat8 or amct_ops backend is required")
```

fallback 中发生的实际数据流是：

```python
codes = encode_to_hifloat8(work_tensor)  # NPU uint8，每个元素是真实 HiFloat8 code
quantized = decode_from_hifloat8(codes, work_dtype)
```

是调用编译后的 Ascend C
算子在 NPU 上查 LUT。`quantized` 保留量化后的离散值，但由于当前 PyTorch Linear 不能
直接使用 `uint8` 编码，评测前必须解码回 BF16。

### 3.3 三种 AMCT 算法

| 算法 | 权重处理 | 激活处理 | 校准 |
| --- | --- | --- | --- |
| Cast | scale 后 HiFloat8 roundtrip | 直接 HiFloat8 roundtrip | 无 |
| Quantile | 固定权重 scale 后 roundtrip | batch absmax 的 0.99/0.01 EMA scale | 有 |
| OFMR | 11 个 `2^ec, ec in [-5,5]` 候选 | 以输出 MSE 选择 scale | 有 |

量化态模块缓存的是解码后的 BF16 权重，并继续调用 BF16 Linear。

---

## 4. 环境搭建

### 4.1 构建 AMCT 和自定义算子

```bash
source /home/developer/Ascend/cann-9.0.0/set_env.sh
git clone --branch master --single-branch https://gitcode.com/cann/amct.git amct-master
cd amct-master
git checkout bd13cdd9357d9af9578ffef1a9e9cd86bef4ea1f

bash amct_ops/ops_build.sh --soc ascend910_93 hifloat8_cast
bash build.sh --torch

python3 -m venv --system-site-packages ../.venv-amct-qwen
../.venv-amct-qwen/bin/python -m pip install \
  'transformers==4.51.0' 'datasets==4.8.4' accelerate einops fastparquet
../.venv-amct-qwen/bin/python -m pip install --no-deps \
  build_out/amct_pytorch-1.1.0-py3-none-linux_aarch64.tar.gz \
  amct_ops/dist/amct_ops-1.0.0-cp311-cp311-linux_aarch64.whl
../.venv-amct-qwen/bin/python -m pip check
```

```bash
# 1. 直接从 staging 运行 hifloat8_cast 官方 NPU 测试
cd /mnt/workspace/amct-master
PYTHONPATH=amct_ops/staging \
python3 -m unittest -v tests.amct_ops.test_hifloat8_cast

# 2. 单独探测 native：预期在本环境失败，并保留 161002 错误
python3 tests/amct_ops/run_hifloat8_dtype_smoke.py \
  --backend native --device npu:0

# 3. 强制 amct_ops 与自动路由
PYTHONPATH=amct_ops/staging \
/mnt/workspace/.venv-amct/bin/python \
  tests/amct_ops/run_hifloat8_dtype_smoke.py \
  --backend amct_ops --device npu:0
PYTHONPATH=amct_ops/staging \
/mnt/workspace/.venv-amct/bin/python \
  tests/amct_ops/run_hifloat8_dtype_smoke.py \
  --backend auto --device npu:0

# 4. 检查 AMCT dtype 与 fallback 单元逻辑
/mnt/workspace/.venv-amct/bin/python -m pytest -q \
  tests/unit_test/quantization/test_hifp_impl.py \
  tests/unit_test/quantization/test_dtypes.py
```

> **结果分别是：** 官方算子测试 `10/10`；native 失败；`amct_ops` 和 `auto` 均通过且 auto
> 实际选择 `amct_ops`；dtype 测试 `69 passed, 1 skipped`。
>
> 编译中出现 Kineto 缺失告警，为正常现象。

最后，我先用两层 BF16 MLP 做三算法benchmark。

```bash
cd experiment/task-book/amct_experience_sunshine750
python tools/smoke_amct_algorithms.py \
  --output /tmp/smoke_amct_algorithms.json
```

> Cast、Quantile、OFMR 均完成 NPU 前向，输出全为有限值；对应最大绝对误差分别为
> 0.02637、0.02734、0.02832。

### 4.2 模型

- 模型：Qwen/Qwen3-0.6B。
- revision：`c1899de289a04d12100db370d81485cdf75e47ca`。
- 参数量：596,049,920。
- 加载 dtype：BF16。
- 权重许可：Apache 2.0。

主站在本容器超时，hf-mirror 的 Xet CAS 返回 401，因此
权重经 ModelScope 直链下载。权重 SHA 同时匹配 Hugging Face LFS OID 和 ModelScope
`X-Linked-Etag`。

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 hf download \
  Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --exclude model.safetensors \
  --local-dir /data/Qwen3-0.6B

curl --fail --location --retry 5 \
  --output /data/Qwen3-0.6B/model.safetensors \
  https://modelscope.cn/models/Qwen/Qwen3-0.6B/resolve/master/model.safetensors
```

### 4.3 数据集

- 数据集：Salesforce/Wikitext，配置 `wikitext-2-raw-v1`。
- revision：`b08601e04326c79dfdd32d625aee71d232d685c3`。
- test：4,358 行，Qwen tokenizer 后 299,078 tokens。
- train：36,718 行，Qwen tokenizer 后 2,518,423 tokens。
- 许可：CC BY-SA 3.0 / GFDL。

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 hf download \
  Salesforce/wikitext --repo-type dataset \
  --revision b08601e04326c79dfdd32d625aee71d232d685c3 \
  --include 'wikitext-2-raw-v1/*' \
  --local-dir /data/wikitext

python - <<'PY'
from pathlib import Path
import pandas as pd
base = Path('/data/wikitext/wikitext-2-raw-v1')
for split in ('train', 'validation', 'test'):
    source = base / f'{split}-00000-of-00001.parquet'
    values = pd.read_parquet(source, engine='fastparquet')['text'].tolist()
    (base / f'{split}.txt').write_text('\n\n'.join(values), encoding='utf-8')
PY
```

这里实际遇到的错误是 PyArrow 25 读取由 Arrow 12 生成的 Parquet dictionary page 时越界。
我改用同样基于 Parquet 元数据读取的 `fastparquet`，再将每行`text` 按 `"\n\n"` 连接成固定 TXT。

---

## 5. 实验

### 5.1 PPL

- test token 按连续 4096 tokens 分段，丢弃最后不足 4096 的 70 tokens。
- 共 73 段，每段预测后 4095 tokens，实际分母为 298,935 tokens。
- logits 转 FP32 后计算 sum cross entropy。
- `PPL = exp(total_nll / predicted_tokens)`。
- baseline 与每种算法都从同一 checkpoint 重新加载。
- `attn_implementation='eager'`，`use_cache=False`，seed 42。

我选择 PPL 而不是只比较若干权重的平均误差，是因为量化误差经过 28 层网络传播后，
最终对语言建模能力的影响才是验收真正关心的结果。另一方面，为了控制计算成本，我没有
采用滑窗重叠评测，而是采用连续、互不重叠的 4096-token segment。分母使用每段真正被
预测的 4095 个 token，不把第一个无前文 token 错算进去。

`quantize.py` 的执行顺序是：参数和资产校验 -> 固定随机种子 -> 记录环境与哈希 -> 加载
tokenizer/model -> 统计 Linear 范围 -> `amct.quantize()` -> 校准（如需要） -> PPL ->
可选 `amct.convert()` 探测 -> 分阶段写 JSON。

### 5.2 方案对比

| 方案 | Weight | Input | skip_layers | 校准 | 量化 Linear |
| --- | --- | --- | --- | --- | ---: |
| Cast | channel | tensor | `lm_head` | 无 | 196 |
| Quantile | tensor | tensor | `lm_head` | 32 x 512 tokens | 196 |
| OFMR | tensor | tensor | 无 | 1 x 512 tokens | 197 |

Quantile 将 `batch_num` 显式设为 32，使 0.99/0.01 EMA 真正跨 batch 更新。OFMR 的单
batch 校准已遍历每层 11 个 scale 候选，成本显著高于 Quantile。

### 5.3 对比

控制组统一为 weight/input tensor，统一跳过 `lm_head`。Quantile 官方配置已与控制组
完全一致，因此复用同一结果；Cast 只改变 weight channel -> tensor，OFMR 只增加跳过
`lm_head`。

增加控制组的原因是我在源码审阅时发现三份“官方默认配置”并不只改变算法。如果把默认
结果直接命名成纯算法优劣，Cast 的 per-channel 优势和 OFMR 多量化一个 `lm_head` 的
影响都会影响结论。

---

## 6. 量化

```bash
PY=/path/to/.venv-amct-qwen/bin/python
SCRIPT=experiment/task-book/amct_experience_sunshine750/quantize.py
MODEL=/data/Qwen3-0.6B
DATA=/data/wikitext/wikitext-2-raw-v1
COMMON="--model-path $MODEL --dataset-path $DATA --device npu:0 --seq-len 4096"

$PY $SCRIPT $COMMON --algorithm baseline \
  --output /tmp/hif8_baseline.json
$PY $SCRIPT $COMMON --algorithm cast --profile official \
  --output /tmp/hif8_cast.json
$PY $SCRIPT $COMMON --algorithm quantile --profile official \
  --calibration-batches 32 --calibration-seq-len 512 \
  --output /tmp/hif8_quantile.json
$PY $SCRIPT $COMMON --algorithm ofmr --profile official \
  --calibration-batches 1 --calibration-seq-len 512 \
  --output /tmp/hif8_ofmr.json
```

```bash
$PY $SCRIPT $COMMON --algorithm cast --profile controlled \
  --output /tmp/hif8_cast_controlled.json
$PY $SCRIPT $COMMON --algorithm ofmr --profile controlled \
  --calibration-batches 1 --calibration-seq-len 512 \
  --output /tmp/hif8_ofmr_controlled.json
$PY $SCRIPT --model-path $MODEL --dataset-path $DATA --device npu:0 \
  --seq-len 128 --max-eval-segments 1 --algorithm cast --try-convert \
  --output /tmp/hif8_convert_probe.json
```

---

## 7. 结果

![Qwen3-0.6B HiFloat8 精度对比](figures/ppl_accuracy_comparison.png)

### 7.1 官方默认方案

| 模型/方案 | PPL | 绝对变化 | 相对变化 | 校准 s | 评测 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 baseline | 19.1651208 | 0 | 0 | - | 23.07 |
| HiFloat8 Cast | 19.4163551 | +0.2512343 | +1.3109% | 0 | 159.64 |
| HiFloat8 Quantile | **19.2813812** | **+0.1162604** | **+0.6066%** | 1.88 | 162.64 |
| HiFloat8 OFMR | 19.7913465 | +0.6262257 | +3.2675% | 15.91 | 166.83 |

### 7.2 控制变量结果

| 算法 | PPL | 相对 BF16 | 相对对应官方配置 |
| --- | ---: | ---: | ---: |
| Cast，weight tensor | 19.4435107 | +0.2783899 | +0.0271556 |
| Quantile，复用官方结果 | **19.2813812** | **+0.1162604** | 0 |
| OFMR，跳过 lm_head | 19.5446361 | +0.3795153 | -0.2467104 |

1. Quantile 在默认组和控制组中都是精度最佳方案。
2. Cast 的 per-channel weight 策略贡献约 0.027 PPL，默认粒度差异确实会混入算法对比。
3. OFMR 跳过 `lm_head` 可改善约 0.247 PPL，默认量化范围是显著混杂变量。
4. 本模型上增加算法复杂度并不自动带来更高精度；OFMR 单 batch 校准仍弱于 Quantile。

---

## 8. 性能和存储结果

### 8.1 HiFloat8 cast 算子 benchmark

![HiFloat8 cast 算子吞吐曲线](figures/cast_operator_benchmark.png)

**方法：** 10 次预热、100 次测量，每次调用后 NPU synchronize，吞吐为十进制 MB/s。

```bash
cd experiment/task-book/amct_experience_sunshine750
python tools/benchmark_hifloat8_cast.py \
  --output /tmp/benchmark_hifloat8_cast.json
```

FP16/BF16、1K/4K/16K/64K/256K/1M/4M/16M 八种，同时记录首次
调用、encode、decode、roundtrip 以及连续/转置非连续输入。

| dtype | 元素数 | Encode ms | Encode MB/s | Decode ms | Decode MB/s | Roundtrip ms | 有效 MB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | 1K | 0.116 | 26.5 | 0.120 | 25.7 | 0.196 | 20.9 |
| FP16 | 1M | 0.930 | 3383.9 | 0.929 | 3387.1 | 1.804 | 2324.4 |
| FP16 | 16M | 13.325 | 3777.2 | 13.234 | 3803.2 | 26.513 | 2531.1 |
| BF16 | 1K | 0.119 | 25.8 | 0.122 | 25.2 | 0.204 | 20.1 |
| BF16 | 1M | 0.920 | 3418.3 | 0.919 | 3424.6 | 1.795 | 2336.4 |
| BF16 | 16M | 13.328 | 3776.5 | 13.237 | 3802.2 | 26.515 | 2530.9 |

roundtrip 有效吞吐按原输入 2 B + 最终输出 2 B，即 4 B/元素计算；两个 kernel 实际
逻辑搬运为 6 B/元素。FP16 首次 1K encode 为 2.77 ms，LUT 热态均值为 0.116 ms。
2048 x 2048 转置非连续输入在本机比连续输入慢约 1.5%。

### 8.2 模型

![量化态模拟的耗时和显存不能代表原生部署](figures/fake_quant_cost_and_memory.png)

| 项目 | 数值 | 性质 |
| --- | ---: | --- |
| 被量化 Linear | 196 | 实测模块替换 |
| 被量化权重元素 | 440,401,920 | 实测结构统计 |
| BF16 原始 payload | 840 MiB | 理论裸权重数据 |
| HiFloat8 payload | 420 MiB | 理论裸权重数据，不含 scale/metadata |
| payload 降低 | 50% | 格式理论值 |

> 量化峰值显存为 baseline 6.93 GiB、Quantile 7.75 GiB，且 Quantile 评测明显慢于
> BF16。这是因为 fake-quant 保留 BF16 权重并增加编解码。

---

## 9. 兼容性矩阵

| 路径 | 结果 | 说明 |
| --- | --- | --- |
| 最小 NPU FP16 benchmark | 通过 | runtime/driver 基础链路正常 |
| hifloat8_cast 构建 | 通过 | 生成 amct_ops wheel |
| 官方算子测试 | 10/10 通过 | FP16/BF16、特殊值、随机 roundtrip |
| native HiFloat8 cast | 失败 | `aclnnCast` 161002 |
| amct_ops backend | 通过 | 硬件路径已验证 |
| auto backend | 通过 | 实际选择 amct_ops |
| HiFloat dtype | 69 passed, 1 skipped | dtype 与 fallback 逻辑通过 |
| Cast/Quantile/OFMR fake quant | 通过 | Qwen3 全量 PPL 已完成 |
| `amct.convert()` | 失败 | `aclnnQuantize` 不支持 `DT_HIFLOAT8` |
| current device=0、input=npu:1 | 通过 | 基础跨 current/input device 场景通过 |

---

## 10. 问题

### 10.1 native dtype 暴露与实际内核能力不一致

- 实际结果：`torch_npu.hifloat8` 存在，但 native cast 和 quantize 均失败。
- convert 错误：`161002/EZ1001`，`DT_HIFLOAT8` 不在支持列表
  `[DT_INT8, DT_UINT8, DT_INT32]`。

### 10.2 ：对称权重 scale 使用正最大值而非绝对最大值

**源码位置：**

- `amct_pytorch/classic/quantize_op/utils.py:149-168`
- `amct_pytorch/classic/quantize_op/quantile_module.py:98-112`

当前 Cast/Quantile 权重路径使用 `weight.max()/16`。对称量化应使用
`weight.abs().amax()/16` 或 `max(abs(min), abs(max))/16`。全负通道会得到负 scale，
随后被 `process_scale()` 替换为 1.0。

| 指标 | 当前 max/16 | absmax/16 对照 |
| --- | ---: | ---: |
| scale | `[0.0018768, 1.0]` | `[0.0625, 7.5]` |
| 最大绝对误差 | 8.0 | 2.5 |
| 平均绝对误差 | 2.255 | 0.625 |

```bash
cd experiment/task-book/amct_experience_sunshine750
python tools/reproduce_cast_weight_scale.py \
  --output /tmp/reproduce_cast_weight_scale.json
```

我从源码看到 `weight.max()/16`，再专门构造一个
负向绝对值大于正向绝对值的通道和一个全负通道，这样可以看到错误。

### 10.3 仓库统一 Transformers pin 与 Qwen3 环境 ABI 冲突

根依赖固定 Transformers 5.12.1。在本机 torch 2.7.1 + torchaudio 2.11 环境中，加载
Qwen3 会触发 torchaudio 导入并报 `undefined symbol: torch_library_impl`。固定 4.51.0
后通过。
