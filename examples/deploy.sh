

python amct_pytorch/cli/llm/deploy.py \
  --model /path/to/model \
  --model_name qwen3_5 \
  --device npu:0 \
  --granularity block \
  --quant_target mlp attn-linear \
  --quant_dtype int \
  --bit_config amct_pytorch/configs/w8a8.yaml \
  --output_dir ./deploy_out

