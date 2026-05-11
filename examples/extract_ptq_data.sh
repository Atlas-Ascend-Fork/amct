

python amct_pytorch/cli/llm/extract_ptq_data.py \
  --model /path/to/model \
  --model_name qwen3_5 \
  --device npu:0 \
  --granularity block \
  --quant_target mlp \
  --data_dir /path/to/data \
  --output_dir ./outputs \
  --nsamples 128