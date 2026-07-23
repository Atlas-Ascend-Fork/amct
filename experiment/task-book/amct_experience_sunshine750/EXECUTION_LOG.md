# HiFloat8 体验任务记录

---

## Step 1：检查系统资源

```bash
uname -a
cat /etc/os-release
lscpu
free -h
df -h /mnt/workspace
```

> 该系统：Ubuntu 20.04.5、aarch64、40 CPU 核、229 GiB 内存（196 GiB 可用）。

---

## Step 2：检查 NPU、驱动和设备节点

```bash
ls -l /dev/davinci* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
npu-smi info
npu-smi info -m
cat /usr/local/Ascend/driver/version.info
```

> **结果：** 两颗 NPU ，节点为 davinci10/11。

---

## Step 3：CANN 和工具链

```bash
cat /home/developer/Ascend/cann-9.0.0/aarch64-linux/include/version/cann_version.h
cat /home/developer/Ascend/cann-9.0.0/share/info/runtime/version.info
cat /home/developer/Ascend/cann-9.0.0/opp/version.info
python3 --version
cmake --version
gcc --version
```

> **结果：** CANN runtime/ops 为 9.0.0，Python 3.11.4，CMake 3.20.5，GCC 9.4.0。

---

## Step 4：检查 Python 和 NPU

```bash
python3 -m pip list
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
python3 -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
```

> **结果：** torch 2.7.1+cpu、torch_npu 2.7.1.post4，NPU available，count=2；
> 缺少：AMCT、amct_ops、transformers、datasets。

---

## Step 5：NPU 运算benchmark

```bash
python3 -c "import torch, torch_npu; x=torch.tensor([1.,-1.], dtype=torch.float16, device='npu:0'); print((x*2).cpu())"
```

> **结果：** 得到 `[2, -2]`，说明当前我配置的没啥问题。

---

## Step 6：发现官方分支的哈希值不同

```bash
git ls-remote --heads https://gitcode.com/cann/amct.git
git clone --depth 1 --branch feature/community-tasks --single-branch \
  https://gitcode.com/cann/amct.git amct-community-tasks
git clone --depth 1 --branch master --single-branch \
  https://gitcode.com/cann/amct.git amct-master
```

> **结果：**
>
> - master：`bd13cdd...`，包含源码和 HiFloat8 新版本的提交。
> - master 已新增 native 探测失败后的 amct_ops fallback，之前的那个线性无法适配。

---

## Step 7：编译 hifloat8_cast

> **路径：** `/mnt/workspace/amct-master`

```bash
bash amct_ops/ops_build.sh --soc ascend910_93 hifloat8_cast
```

> **结果：** 出现报错，Kineto 缺失告警与仓库中的已知告警一致，不影响链接。

---

## Step 8：运行官方算子benchmark

```bash
PYTHONPATH=amct_ops/staging python3 -m unittest -v tests.amct_ops.test_hifloat8_cast
```

> **结果：** 通过，耗时 1.064 秒。

---

## Step 9：HiFloat8检查

先直接调用 `torch_npu.npu_dtype_cast`，再运行官方 smoke：

```bash
python3 tests/amct_ops/run_hifloat8_dtype_smoke.py --backend native --device npu:0
```

> **结果：** 失败，`aclnnCast` 错误 161002，`DT_HIFLOAT8` 不在 dtype 支持列表。

之后进一步测试：

保持 current device=0，在 `npu:1` 创建输入并调用 amct_ops encode/decode。

> **结果：** 输出仍在 `npu:1`，`[1,-1,0.5,2]` 。说明这个确实是没有显式支持这个Hifloat8的（在9.0.0这个版本之下）

---

## Step 10：创建 Python 环境

```bash
python3 -m venv --system-site-packages /mnt/workspace/.venv-amct
/mnt/workspace/.venv-amct/bin/python -m pip install -r requirements.txt
```

> **结果：** 成功安装了 `transformers 5.12.1 + datasets 4.8.4` 。

---

## Step 11：运行 HiFloat dtype

```bash
.venv-amct/bin/python -m pytest -q \
  tests/unit_test/quantization/test_hifp_impl.py \
  tests/unit_test/quantization/test_dtypes.py
```

> **结果：** 69 passed，1 skipped。

---

## Step 12：构建并安装 AMCT

```bash
bash build.sh --torch
.venv-amct/bin/python -m pip install --no-build-isolation \
  build_out/amct_pytorch-1.1.0-py3-none-linux_aarch64.tar.gz \
  amct_ops/dist/amct_ops-1.0.0-cp311-cp311-linux_aarch64.whl
.venv-amct/bin/python -m pip check
```

> **结果：** AMCT 1.1.0 构建成功。

---

## Step 13：运行 cast benchmark

```bash
.venv-amct/bin/python hifloat8_report_work/benchmark_hifloat8_cast.py \
  --output hifloat8_report_work/benchmark_hifloat8_cast.json
```

> **结果：** FP16/BF16、1K 到 16M、encode/decode/roundtrip 共 16 组。16M
> 编码/解码约 3.78/3.80 GB/s，roundtrip 有效吞吐约 2.53 GB/s。

---

## Step 14：查看 Cast/Quantile/OFMR 实现

详细阅读内置配置、三种 quant module、deploy module、官方 Qwen 示例和 HiFloat8 文档。

> **结果：**
>
> - Cast 无校准，激活直接 roundtrip，权重缩放。
> - Quantile 按 batch 最大绝对值做 EMA，但默认 batch_num=1。
> - OFMR 对 `EC_CAND=(-5..5)` 的 11 个 2 的幂 scale 计算输出 MSE 并择优选择。
> - Cast 默认 weight per-channel，Quantile/OFMR 默认 per-tensor；OFMR 默认不跳过 lm_head。
> - convert 使用 native quantize/matmul，不走 amct_ops fallback。

---

## Step 15：下载公开模型与数据

```bash
df -h /mnt/workspace
/mnt/workspace/.venv-amct/bin/hf --help
curl -I -L --max-time 20 \
  https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json
curl -I -L --max-time 25 \
  https://hf-mirror.com/Qwen/Qwen3-0.6B/resolve/main/config.json
curl -I -L --max-time 25 \
  https://modelscope.cn/models/Qwen/Qwen3-0.6B/resolve/master/config.json
```

> **结果：** Hugging Face 主站超时；hf-mirror 返回错误，ModelScope 可用。

---

## Step 16：下载 Qwen3-0.6B

```bash
HF_ENDPOINT=https://hf-mirror.com \
/mnt/workspace/.venv-amct/bin/hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir /mnt/workspace/hifloat8_report_work/assets/Qwen3-0.6B
```

禁用 Xet 后普通镜像权重连接长时间停在 10 MiB，改从 ModelScope 直链下载：

```bash
kill 53151
cd /mnt/workspace/hifloat8_report_work/assets/Qwen3-0.6B
curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
  --output model.safetensors.part \
  https://modelscope.cn/models/Qwen/Qwen3-0.6B/resolve/master/model.safetensors
sha256sum model.safetensors.part
mv model.safetensors.part model.safetensors
```

---

## Step 17：下载并规范化 Wikitext-2

第一次整库下载误包含 Wikitext-103，终止实际 PID 55612，随后只拉取目标配置：

```bash
kill 55612
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
/mnt/workspace/.venv-amct/bin/hf download Salesforce/wikitext \
  --repo-type dataset \
  --revision b08601e04326c79dfdd32d625aee71d232d685c3 \
  --include 'wikitext-2-raw-v1/*' \
  --local-dir /mnt/workspace/hifloat8_report_work/assets/wikitext \
  --max-workers 2
```

> **结果：** 下载到 train/validation/test 三份 Parquet。但 PyArrow 25 读取 Arrow 12 写入的字典页时出现 `Index not in dictionary bounds`。

---

## Step 18：建立 Qwen3 环境

仓库根依赖固定 `transformers==5.12.1`，首次加载因系统 torchaudio 2.11 与 torch
2.7.1 ABI 不匹配而失败。模型配置和仓库 Qwen3 样例均指向 4.51.0，因此另建环境：

```bash
python3 -m venv --system-site-packages /mnt/workspace/.venv-amct-qwen
/mnt/workspace/.venv-amct-qwen/bin/python -m pip install \
  'transformers==4.51.0' 'datasets==4.8.4' accelerate
/mnt/workspace/.venv-amct-qwen/bin/python -m pip install --no-deps \
  /mnt/workspace/amct-master/build_out/amct_pytorch-1.1.0-py3-none-linux_aarch64.tar.gz \
  /mnt/workspace/amct-master/amct_ops/dist/amct_ops-1.0.0-cp311-cp311-linux_aarch64.whl
/mnt/workspace/.venv-amct-qwen/bin/python -m pip install einops
/mnt/workspace/.venv-amct-qwen/bin/python -m pip check
```

> **结果：** Qwen3 BF16 128-token NPU smoke 输出全部有限，峰值约
> 1.20 GiB。随后安装独立 Parquet 引擎，按官方样例的 `"\n\n".join(text)` ：

```bash
/mnt/workspace/.venv-amct-qwen/bin/python -m pip install fastparquet
/mnt/workspace/.venv-amct-qwen/bin/python - <<'PY'
from pathlib import Path
import pandas as pd
base = Path('/mnt/workspace/hifloat8_report_work/assets/wikitext/wikitext-2-raw-v1')
for split in ('train', 'validation', 'test'):
    source = base / f'{split}-00000-of-00001.parquet'
    values = pd.read_parquet(source, engine='fastparquet')['text'].tolist()
    (base / f'{split}.txt').write_text('\n\n'.join(values), encoding='utf-8')
PY
sha256sum /mnt/workspace/hifloat8_report_work/assets/wikitext/wikitext-2-raw-v1/*.txt
```

> **结果：** Qwen tokenizer 分别得到
> 2,518,423/299,078 tokens。

---

## Step 19：实现并验证量化全过程

新增 `delivery_draft/quantize.py`，实现本地模型/数据哈希、BF16 PPL、Cast/Quantile/
OFMR、固定校准采样、官方/控制配置、显存计量、convert 探测和 JSON 结果。

```bash
/mnt/workspace/.venv-amct-qwen/bin/python -m py_compile \
  /mnt/workspace/hifloat8_report_work/delivery_draft/quantize.py
/mnt/workspace/.venv-amct/bin/python -m ruff check \
  /mnt/workspace/hifloat8_report_work/delivery_draft/quantize.py
```

随后分别用 `seq_len=128, max_eval_segments=1` 跑 baseline、Cast、2-batch
Quantile 和 1-batch OFMR smoke。

---

## Step 20：运行 PPL

> **公共参数：** Qwen3-0.6B BF16、Wikitext-2 test、`seq_len=4096`、73 段、seed 42。

```bash
cd /mnt/workspace/hifloat8_report_work
/mnt/workspace/.venv-amct-qwen/bin/python delivery_draft/quantize.py \
  --model-path assets/Qwen3-0.6B --dataset-path assets/wikitext/wikitext-2-raw-v1 \
  --device npu:0 --seq-len 4096 --algorithm baseline \
  --output results/baseline_official_full.json
/mnt/workspace/.venv-amct-qwen/bin/python delivery_draft/quantize.py \
  --model-path assets/Qwen3-0.6B --dataset-path assets/wikitext/wikitext-2-raw-v1 \
  --device npu:0 --seq-len 4096 --algorithm cast --profile official \
  --output results/cast_official_full.json
/mnt/workspace/.venv-amct-qwen/bin/python delivery_draft/quantize.py \
  --model-path assets/Qwen3-0.6B --dataset-path assets/wikitext/wikitext-2-raw-v1 \
  --device npu:0 --seq-len 4096 --algorithm quantile --profile official \
  --calibration-batches 32 --calibration-seq-len 512 \
  --output results/quantile_official_full.json
/mnt/workspace/.venv-amct-qwen/bin/python delivery_draft/quantize.py \
  --model-path assets/Qwen3-0.6B --dataset-path assets/wikitext/wikitext-2-raw-v1 \
  --device npu:0 --seq-len 4096 --algorithm ofmr --profile official \
  --calibration-batches 1 --calibration-seq-len 512 \
  --output results/ofmr_official_full.json
/mnt/workspace/.venv-amct-qwen/bin/python delivery_draft/quantize.py \
  --model-path assets/Qwen3-0.6B --dataset-path assets/wikitext/wikitext-2-raw-v1 \
  --device npu:0 --seq-len 4096 --algorithm cast --profile controlled \
  --output results/cast_controlled_full.json
/mnt/workspace/.venv-amct-qwen/bin/python delivery_draft/quantize.py \
  --model-path assets/Qwen3-0.6B --dataset-path assets/wikitext/wikitext-2-raw-v1 \
  --device npu:0 --seq-len 4096 --algorithm ofmr --profile controlled \
  --calibration-batches 1 --calibration-seq-len 512 \
  --output results/ofmr_controlled_full.json
```

> **结果：** BF16/官方 Cast/官方 Quantile/官方 OFMR PPL 分别为 19.16512、19.41636、
> 19.28138、19.79135， Cast/OFMR 为 19.44351/19.54464。Quantile 最优，
> 相对 BF16 绝对增量 0.11626。
