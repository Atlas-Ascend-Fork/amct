# model quantization sample

## 🚀Latest News
- [2025/12] DeepSeek-V3.2已支持逐block量化推理

## 🎉概述
experience中存放LLM量化和推理的典型模型sample，不依赖框架，通过高阶PTQ量化算法，得到接近bf16的量化模型精度

## 📖目录结构说明
```
├── docs                                        # 文档目录
|  ├── models                                   # 模型文档目录
|  |  ├── deepseek-v3.2                         # DeepSeek-V3.2相关文档
|  |  └── ...
├── cores                                       # 核心算法目录
|  ├── calibrator                               # 逐block量化学习接口
|  ├── models                                   # Qwen3-MoE的模型脚本及执行配置
|  |  ├── deepseek-v3.2                         # DeepSeek-V3.2相关模型定义
|  ├── quantization                             # 量化层相关定义
|  ├── utils                                    # 通用接口
├── pp                                          # 算子目录
|  ├── forward                                  # 多卡串行推理
|  ├── run_pp_wiki.py                           # 计算wikitext ppl
└── eval.py                                     # wikitext精度计算
└── extract_calib_data.py                       # dump逐block数据
└── main.py                                     # 逐block学习
└── README.md
└── ...
```

## 📝使用说明
我们提供了对应的scrip在`./scripts/`，示例如下：
### 数据提取
```
python3 extract_calib_data.py --model $model_path --output_dir $output_dir
```
### 训练
逐block-C8训练
```
python ./main.py \
 --model $model_path \
 --w_bits 8 --a_bits 8 \
 --q_bits 8 --k_bits 8 --v_bits 8 \
 --cali_bsz 1 --epoch 25 --base_lr 1e-2 \
 --lwc --lac \
 --cls c8 \
 --output_dir $output_path --data_dir $data_path \
 --start_block_idx $start --end_block_idx $end --train_mode mla --dev 0
```
逐expert训练
```
# 根据w_bits切换A8W8或A8W4
python ./main.py \
 --model $model_path \
 --w_bits 8 --a_bits 8 \
 --q_bits 8 --k_bits 8 --v_bits 8 \
 --cali_bsz 1 --epoch 25 --base_lr 1e-2 \
 --lwc --lac \
 --cls c8 \
 --output_dir $output_path --data_dir $data_path \
 --start_block_idx $start --end_block_idx $end --train_mode moe --dev 0
```
### 测试
```
python3 ./eval_ppl.py \
    --a_bits 8 \
    --w_bits 8 \
    --seq_len 4096 \
    --cls c8 \
    --model $model_path \
    --train_mode block \
    --output_dir $output_path \
    --wikitext_final_out $wikitext_out \
    --lac --lwc \
    --start_block_idx 0 --end_block_idx 61 \
    --mla_param_dir $mla_param_dir \
    --moe_param_dir $moe_param_dir
```
### 精度
**量化模型精度表现**

| 模型 | PPL    |
| ---- |--------|
| DeepSeek-V3.2-BF16 | 2.9987 |
| DeepSeek-V3.2-Exp-W8A8C8 | 3.0881 |
| DeepSeek-V3.2-Exp-W4A8C8 | 3.2320 |
### 主函数参数说明
#### eval.py
- **group**：将所有block分为group组，分组并行执行
- **begin**：block序号起始，一般为0
- **end**：block序号末尾，如在DeepSeek-V3.2中为60
- **args.seq_len**：每段文本的长度
- **args.output_dir**：输出保存路径
- **num_npus**：所使用的NPU卡数量，默认为当前窗口下可见的所有NPU卡，单卡显存要求64G
#### main.py
- **args.data_dir**：dump数据的保存路径
- **train_mode**：选择mla/moe训练
- **model_path**：模型文件保存路径
- **cls**：可选为c8和bf16，在训练moe时请选择为bf16，训练mla时请选择为c8
#### deploy.py
- **input_weight_path**：要被转换的权重路径（FP8/BF16）
- **output_weight_path**：转换后权重保存路径
- **quant_type**：量化权重类型（目前支持bfloat16, w8a8c16, w8a8c8, w4a8c16, w4a8c8）
- **clip**：训练时是否做了clip
- **mla_param_path**：MLA训练结果保存路径
- **moe_param_path**：MOE训练结果保存路径