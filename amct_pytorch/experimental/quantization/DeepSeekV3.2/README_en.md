# model quantization sample

## 🚀Latest News
- [2025/12] DeepSeek-V3.2 now supports block-by-block quantization inference

## 🎉Overview
The experimental directory contains typical model samples for LLM quantization and inference. It is framework-independent and achieves quantization model accuracy close to bf16 through advanced PTQ quantization algorithms.

## 📖Directory Structure Description
```
├── docs                                        # Documentation directory
|  ├── models                                   # Model documentation directory
|  |  ├── deepseek-v3.2                         # DeepSeek-V3.2 related documentation
|  |  └── ...
├── cores                                       # Core algorithm directory
|  ├── calibrator                               # Block-by-block quantization learning interface
|  ├── models                                   # Qwen3-MoE model scripts and execution configurations
|  |  ├── deepseek-v3.2                         # DeepSeek-V3.2 related model definitions
|  ├── quantization                             # Quantization layer related definitions
|  ├── utils                                    # Common interfaces
├── pp                                          # Operator directory
|  ├── forward                                  # Multi-card serial inference
|  ├── run_pp_wiki.py                           # Compute wikitext ppl
└── eval.py                                     # wikitext accuracy calculation
└── extract_calib_data.py                       # dump block-by-block data
└── main.py                                     # Block-by-block learning
└── deploy.py                                   # Generate quantization model
└── README.md
└── ...
```

## 📝Usage Instructions
We provide corresponding scripts in `./scripts/`. Examples are as follows:

During the training phase, modify `w_bits`, `a_bits`, `q_bits`, `k_bits`, and `v_bits` according to actual needs. For C8 training, ensure that cls passes c8; otherwise, the MLA part training parameters will have no gradients. When training MoE, please adjust cls to bf16.

During the testing phase, modify `w_bits`, `a_bits`, `q_bits`, `k_bits`, and `v_bits` according to actual needs. Also modify train_mode, which is divided into `mla`, `moe`, `block`, and `origin` according to quantizing only MLA, quantizing only MoE, quantizing both MLA+MoE, and not quantizing, respectively.

### Data Extraction
```
python3 extract_calib_data.py --model $model_path --output_dir $output_dir
```
### Training
Block-by-block C8 training
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
Expert-by-expert training
```
# Switch between A8W8 or A8W4 according to w_bits
python ./main.py \
 --model $model_path \
 --w_bits 8 --a_bits 8 \
 --q_bits 8 --k_bits 8 --v_bits 8 \
 --cali_bsz 1 --epoch 25 --base_lr 1e-2 \
 --lwc --lac \
 --cls bf16 \
 --output_dir $output_path --data_dir $data_path \
 --start_block_idx $start --end_block_idx $end --train_mode moe --dev 0
```
### Testing
```
python3 ./eval.py \
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
### Accuracy
**Quantization Model Accuracy Performance**

| Model | PPL    |
| ---- |--------|
| DeepSeek-V3.2-BF16 | 2.9987 |
| DeepSeek-V3.2-Exp-W8A8C8 | 3.0304 |
| DeepSeek-V3.2-Exp-W4A8C8 | 3.2320 |
### Main Function Parameter Description
#### eval.py
- **group**: Divide all blocks into group groups and execute in parallel in groups
- **begin**: Block sequence number start, usually 0
- **end**: Block sequence number end, such as 60 in DeepSeek-V3.2
- **args.seq_len**: Length of each text segment
- **args.output_dir**: Output save path
- **num_npus**: Number of NPU cards used, defaults to all NPU cards visible in the current window. Single card memory requirement is 64G
#### main.py
- **args.data_dir**: Save path for dumped data
- **train_mode**: Select mla/moe training
- **model_path**: Model file save path
- **cls**: Can be c8 or bf16. Please select bf16 when training moe and c8 when training mla
#### deploy.py
- **input_weight_path**: Weight path to be converted (FP8/BF16)
- **output_weight_path**: Converted weight save path
- **quant_type**: Quantized weight type (currently supports bfloat16, w8a8c16, w8a8c8, w4a8c16, w4a8c8)
- **clip**: Whether clip was done during training
- **mla_param_path**: MLA training result save path
- **moe_param_path**: MoE training result save path