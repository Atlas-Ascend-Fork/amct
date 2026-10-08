# Qwen3-0.6B reconstruct/ls: prune_ratio=0.1

本报告记录一次完整的 NPU 实验。运行命令、数据处理协议和计时定义见上级目录 README 的第 8 节。实验未执行额外后处理。

## 剪枝诊断

```text
[prune-diagnose] prunable targets: cnn=0, dense=28, moe=0
  fixed-ratio prune: available (cut 4.4%, forward OK)
  acc binary search: unavailable
  - acc search found no acceptable prune ratio at this tolerance (relax the tolerance).
```

## prune_config.json

```json
{
  "methods": {
    "dense": {
      "name": "reconstruct",
      "kwargs": {
        "prune_ratio": 0.1,
        "recovery": "ls"
      }
    }
  }
}
```

## run_metrics.json

```json
{
  "model": "Qwen3-0.6B",
  "method": "reconstruct",
  "recovery": "ls",
  "mode": "ratio",
  "prune_ratio": 0.1,
  "baseline_ppl": 19.157548904418945,
  "pruned_ppl": 24.783754348754883,
  "post_process_ppl": "N/A",
  "params_before": 596049920,
  "params_after": 569643008,
  "param_cut_ratio": 0.0443031885651457,
  "prune_time_minutes": 0.40711025933366424,
  "calib_prepare_seconds": 7.691561069979798,
  "prune_execute_seconds": 16.735054490040056,
  "diagnose_time_seconds": 65.04378988000099,
  "calib_samples": 32,
  "calib_seq_len": 512,
  "eval_batches": 73,
  "seq_len": 4096,
  "ffn_width": 2765,
  "run_kind": "full_evaluation",
  "save_model_status": "saved_and_validated",
  "dtype": "bfloat16",
  "device": "npu:0"
}
```

## prune_report.json

```json
{
  "backend": "huggingface",
  "params_before": 596049920,
  "params_after": 569643008,
  "events": [
    {
      "stage": "cnn",
      "method": "variance_channel",
      "module": "<none>",
      "detail": "No compatible pruning targets were found"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.0.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.1.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.2.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.3.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.4.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.5.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.6.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.7.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.8.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.9.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.10.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.11.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.12.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.13.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.14.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.15.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.16.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.17.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.18.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.19.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.20.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.21.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.22.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.23.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.24.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.25.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.26.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.27.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    },
    {
      "stage": "moe",
      "method": "activation_count",
      "module": "<none>",
      "detail": "No compatible pruning targets were found"
    }
  ],
  "budget_unreachable": false,
  "prunable_fraction": null,
  "warnings": [],
  "per_layer_sparsity": {
    "model.layers.0.mlp.down_proj": 0.09993489583333337,
    "model.layers.1.mlp.down_proj": 0.09993489583333337,
    "model.layers.2.mlp.down_proj": 0.09993489583333337,
    "model.layers.3.mlp.down_proj": 0.09993489583333337,
    "model.layers.4.mlp.down_proj": 0.09993489583333337,
    "model.layers.5.mlp.down_proj": 0.09993489583333337,
    "model.layers.6.mlp.down_proj": 0.09993489583333337,
    "model.layers.7.mlp.down_proj": 0.09993489583333337,
    "model.layers.8.mlp.down_proj": 0.09993489583333337,
    "model.layers.9.mlp.down_proj": 0.09993489583333337,
    "model.layers.10.mlp.down_proj": 0.09993489583333337,
    "model.layers.11.mlp.down_proj": 0.09993489583333337,
    "model.layers.12.mlp.down_proj": 0.09993489583333337,
    "model.layers.13.mlp.down_proj": 0.09993489583333337,
    "model.layers.14.mlp.down_proj": 0.09993489583333337,
    "model.layers.15.mlp.down_proj": 0.09993489583333337,
    "model.layers.16.mlp.down_proj": 0.09993489583333337,
    "model.layers.17.mlp.down_proj": 0.09993489583333337,
    "model.layers.18.mlp.down_proj": 0.09993489583333337,
    "model.layers.19.mlp.down_proj": 0.09993489583333337,
    "model.layers.20.mlp.down_proj": 0.09993489583333337,
    "model.layers.21.mlp.down_proj": 0.09993489583333337,
    "model.layers.22.mlp.down_proj": 0.09993489583333337,
    "model.layers.23.mlp.down_proj": 0.09993489583333337,
    "model.layers.24.mlp.down_proj": 0.09993489583333337,
    "model.layers.25.mlp.down_proj": 0.09993489583333337,
    "model.layers.26.mlp.down_proj": 0.09993489583333337,
    "model.layers.27.mlp.down_proj": 0.09993489583333337
  },
  "allocation_choice": null
}
```

## reload_validation.json

```json
{
  "status": "passed",
  "state_dict_tensors_checked": 311,
  "all_weights_equal": true,
  "params": 569643008,
  "ffn_widths": [
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765,
    2765
  ],
  "tokenizer_check": "passed",
  "evaluation_blocks": 1,
  "seq_len": 4096,
  "reference_ppl": 21.380943298339844,
  "reloaded_ppl": 21.380943298339844
}
```

## data_protocol.json

```json
{
  "calibration_source": "mit-han-lab/pile-val-backup",
  "calibration_split": "validation",
  "calibration_seed": 42,
  "calibration_samples": 32,
  "calibration_seq_len": 512,
  "calibration_preprocessing": "shared get_pileval/pileval_awq: shuffle seed 42; strip; encode; skip rows longer than seq_len; concatenate; first N full blocks",
  "calibration_token_sha256": "9b53cceebb55c400301dc5a2fb413af43335b88d0dc7b0be1a47ca8c13c00ed8",
  "evaluation_source": "Salesforce/wikitext",
  "evaluation_config": "wikitext-2-raw-v1",
  "evaluation_split": "test",
  "evaluation_preprocessing": "shared get_wiki_inputs: join all rows with double newline; tokenize once; nonoverlapping full blocks; drop remainder",
  "evaluation_blocks": 73,
  "evaluation_seq_len": 4096,
  "evaluation_token_sha256": "9a590bbe468c4971af409387aaac3a7e917bcf6503ef5a8588b501128c84483b",
  "ppl_helper": "utils.eval_wikitext2_ppl"
}
```

## run_environment.json

```json
{
  "python": "3.12.13 (main, Aug  3 2026, 02:40:28) [GCC 11.4.0]",
  "platform": "Linux-5.10.0-216.0.0.115.oe2203sp4.aarch64-aarch64-with-glibc2.35",
  "versions": {
    "torch": "2.7.1+cpu",
    "torch-npu": "2.7.1.post4",
    "transformers": "5.12.1",
    "datasets": "4.8.4",
    "huggingface-hub": "1.33.0",
    "amct-pytorch": "1.1.0"
  },
  "hardware": {
    "name": "Ascend910_9382",
    "visible_count": 1,
    "free_bytes": 65394839552,
    "total_bytes": 65787658240
  },
  "amct_import_path": "/usr/local/python3.12.13/lib/python3.12/site-packages/amct_pytorch/__init__.py",
  "source_hashes": {
    "amct_pytorch.pruning.prune_op.dense_reconstruct": {
      "path": "/usr/local/python3.12.13/lib/python3.12/site-packages/amct_pytorch/pruning/prune_op/dense_reconstruct.py",
      "sha256": "2bd6c46ceae615085253cef0e83e0a513f65e53d198e70c8a0096289785bee53"
    },
    "amct_pytorch.pruning.context": {
      "path": "/usr/local/python3.12.13/lib/python3.12/site-packages/amct_pytorch/pruning/context.py",
      "sha256": "3b5b5891767ccd7c48514980c8b9c8de1c92481f67631baba790a8bd12a3e62f"
    },
    "amct_pytorch.pruning.utils": {
      "path": "/usr/local/python3.12.13/lib/python3.12/site-packages/amct_pytorch/pruning/utils.py",
      "sha256": "0c63633efbdf3f21845806700158d2508eb889ff21da111c1097903245e1a4ac"
    },
    "amct_pytorch.common.datasets.preproc": {
      "path": "/usr/local/python3.12.13/lib/python3.12/site-packages/amct_pytorch/common/datasets/preproc.py",
      "sha256": "458ff7e175d34c57843a743b539d96e3e05ac47661463b6f94a1ffcbde8a3cca"
    },
    "amct_pytorch.common.evaluate.eval_ppl": {
      "path": "/usr/local/python3.12.13/lib/python3.12/site-packages/amct_pytorch/common/evaluate/eval_ppl.py",
      "sha256": "c5184d806897ecc8aa291fc23ea9f4528e0da7e1d7799811fe4c96bff1d5ccbf"
    }
  },
  "sample_sha256": "1b05a501300e1bb4ed8d6a511dc4ee51508d70e2f813888237c15b1586d5390e",
  "helper_sha256": "a5bc1838fedb47e4a553f14fa251864c780d38a2b6e770a947f3d48140b0782d",
  "checkout_commit": "9d54d17e0942052193c827180dc43312c580acc7",
  "environment": {
    "ASCEND_HOME_PATH": "/usr/local/Ascend/cann-9.1.0",
    "ASCEND_OPP_PATH": "/usr/local/Ascend/cann-9.1.0/opp"
  }
}
```

## model_input_sha256.json

```json
{
  "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
  "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
  "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
  "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
  "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
  "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
}
```

## run_config.json

```json
{
  "model_path": "/workspace/models/Qwen3-0.6B",
  "save_dir": "/workspace/task6-records/formal/ratio_0_1",
  "save_model": true,
  "device": "npu:0",
  "prune_ratio": 0.1,
  "calib_samples": 32,
  "calib_seq_len": 512,
  "seq_len": 4096,
  "eval_batches": null,
  "skip_diagnose": false
}
```

## run_status.json

```json
{
  "stage": "complete",
  "status": "complete",
  "error": null
}
```

## 实际运行命令

```bash
/usr/local/python3.12.13/bin/python3 src/run_qwen3_0_6b_pruning_reconstruct_ratio.py --model-path /workspace/models/Qwen3-0.6B --prune-ratio 0.1 --calib-samples 32 --calib-seq-len 512 --seq-len 4096 --device npu:0 --save-model --save-dir /workspace/task6-records/formal/ratio_0_1
```
