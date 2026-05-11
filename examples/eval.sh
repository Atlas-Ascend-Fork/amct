

export ASCEND_RT_VISIBLE_DEVICES=0

# bf16 inference
python amct_pytorch/cli/llm/eval.py \
  --model /path/to/model \
  --model_name qwen3_5 \
  --device npu:0 \
  --granularity block \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml \
  --seq_len 4096

# replace quant module but turn off quant flag, bf16 inference
python amct_pytorch/cli/llm/eval.py \
  --model /path/to/model \
  --model_name qwen3_5 \
  --device npu:0 \
  --granularity block \
  --eval_mode quant \
  --quant_target mlp attn-linear \
  --bit_config amct_pytorch/configs/w8a8.yaml \
  --seq_len 4096


# replace quant module but turn off quant flag, bf16 inference
python amct_pytorch/cli/llm/eval.py \
  --model /path/to/model \
  --model_name qwen3_5 \
  --device npu:0 \
  --granularity block \
  --eval_mode quant \
  --quant_target mlp attn-linear \
  --bit_config amct_pytorch/configs/w8a8.yaml \
  --seq_len 4096
