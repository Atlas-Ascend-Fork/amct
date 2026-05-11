

python amct_pytorch/cli/llm/ptq.py \
  --model /path/to/model \
  --model_name qwen3_5 \
  --device npu:0 \
  --granularity block \
  --quant_target mlp \
  --quant_dtype int \
  --bit_config amct_pytorch/configs/w8a8.yaml \
  --algos lwc lac \
  --output_dir ./outputs
