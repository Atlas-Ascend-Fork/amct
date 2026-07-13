# HiFloat8 量化体验任务报告

GitCode 账号：yyyu222

## 1. 目录结构说明
experiment/task-book/amct_experience_yyyu222/
├── README.md # 本文件，包含环境说明、任务达成情况、体验反馈
└── quantize.py # 量化执行脚本（基于 run_qwen_samples.py 修改）

## 2. 环境等相关说明
基于cannlab的云开发环境进行 HiFloat8 量化的代码的测试（模型加载 → 校准 → 伪量化推理 → PPL 评测），使用qwen3-0.6b模型

## 3. 任务情况

### 3.1 量化输入

| 项目 | 说明 |
| **模型** | Qwen3-0.6B（`qwen/Qwen3-0.6B`），从 ModelScope 下载 |
| **量化算法** | HiFloat8 Quantile 算法 |
| **校准数据集** | `mit-han-lab/pile-val-backup` |
| **评测数据集** | `wikitext2` |
| **评测指标** | 困惑度（Perplexity, PPL） |

### 3.2 量化执行指令

cd /mnt/workspace/gitCode/cann/amct-src-9.1.0/examples/algorithms/quantile/src

python3 quantize.py \
  --model_path=/mnt/workspace/models/models/qwen--Qwen3-0.6B/snapshots/master \
  2>&1 | tee logs/run_qwen_final.log

###  3.3 量化结果

test time taken:  2.0 min  29.080715894699097 s
score:  21.222482681274414

## 4. 测试中遇到的问题及解决办法

1. 数据集无法下载：配置VPN进行相关数据的访问
2. convert接口报错：接口还未完整支持hifloat8类型的数据类型，需要转变成model推理解决
3. 缺失ops库：安装 amct_ops 包（需要先执行 bash ops_build.sh hifloat8_cast 编译算子），为伪量化推理提供 HiFloat8 模拟算子
4. PPL的score结果为nan：数据类型不匹配：使用 torch.float16 加载模型，但 Qwen3 的原始配置为 bfloat16，导致 MLP 层（silu(gate_proj(x)) * up_proj(x)）发生 FP16 乘法溢出，产生 Inf 值。需要将 get_qwen() 中的 torch_dtype 参数改为 torch.bfloat16。

## 5. 优化建议
1. 文档完善：建议在 AMCT 官方文档中明确说明：

2. 运行 Qwen3系列模型时，应使用 torch.bfloat16 而非 torch.float16。

3. amct.convert() 所需的最小 CANN 版本，以及如何检查当前环境是否支持 HiFloat8。

4. 错误提示优化：当 amct.convert() 因算子不支持而失败时，建议给出更友好的提示，引导用户使用伪量化模型进行过渡，并提供 amct_ops 的安装指引。