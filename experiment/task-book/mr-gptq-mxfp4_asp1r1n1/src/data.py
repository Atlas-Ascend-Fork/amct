# -*- coding: UTF-8 -*-
"""数据集加载（自包含）。

封装 WikiText2（评测）与 pile-val（校准）的加载，规避以下环境坑：
- pyarrow 25 读 HF parquet 报 `ArrowInvalid: Index not in dictionary bounds`
  → 改用 fastparquet 引擎读本地 parquet；
- HF 直连被墙 → HF_ENDPOINT=https://hf-mirror.com；
- 大文件走 Xet/CAS 被镜像拒 401 → HF_HUB_DISABLE_XET=1 + allow_patterns 只下所需；
- WikiText 裸名失效 → 规范 id `Salesforce/wikitext`；
- pile-val-backup 非 parquet（`val.jsonl.zst`）→ zstandard 解压读 jsonl。

依赖：pandas, fastparquet, zstandard, huggingface_hub。
"""
import os
import io
import glob
import json

# 国内镜像 + 规避 Xet（在任何 HF 调用前设好）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import pandas as pd


def _load_hf_parquet_fastparquet(repo_id, config, split):
    """用 fastparquet 读 HF parquet，绕开 pyarrow 的字典编码解码 bug。

    只下载所需 config/split 的 parquet（allow_patterns），并禁用 Xet。
    """
    from huggingface_hub import snapshot_download

    if config:
        patterns = [f"{config}/*{split}*.parquet", f"{config}/*.parquet"]
    else:
        patterns = [f"*{split}*.parquet", "*.parquet"]
    local = snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=patterns)

    files = glob.glob(os.path.join(local, "**", "*.parquet"), recursive=True)
    if config:
        files = [f for f in files if config in f] or files
    split_files = [f for f in files if split in os.path.basename(f)]
    files = sorted(split_files or files)
    if not files:
        raise FileNotFoundError(f"no parquet: {repo_id} {config} {split} in {local}")
    return pd.concat(
        [pd.read_parquet(f, engine="fastparquet") for f in files],
        ignore_index=True,
    )


def _load_pile_val_records():
    """读 mit-han-lab/pile-val-backup 的 val.jsonl.zst（zstd 压缩 jsonl）。"""
    import zstandard as zstd
    from huggingface_hub import hf_hub_download

    fp = hf_hub_download(
        repo_id="mit-han-lab/pile-val-backup", repo_type="dataset", filename="val.jsonl.zst"
    )
    records = []
    with open(fp, "rb") as fh:
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_test_dataset(tokenizer):
    """WikiText2 test 全量拼接后 tokenize，返回 [1, total_tokens] 的 input_ids。"""
    df = _load_hf_parquet_fastparquet("Salesforce/wikitext", "wikitext-2-raw-v1", "test")
    enc = tokenizer("\n\n".join(df["text"].tolist()), return_tensors="pt")
    return enc.input_ids


def get_calib_dataset(tokenizer, n_samples=64, block_size=512, max_line_tokens=512, seed=42):
    """从 pile-val 取 n_samples 条（每条 token 数 ≤ max_line_tokens），
    拼接后按 block_size 切块，返回 list[[1, block_size]]。
    """
    import random

    records = _load_pile_val_records()
    random.Random(seed).shuffle(records)

    samples = []
    n_run = 0
    for data in records:
        line = data["text"].strip()
        ids = tokenizer.encode(line)
        if len(ids) > max_line_tokens:
            continue
        sample = torch.tensor([ids])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    cat = torch.cat(samples, dim=1)
    n_split = cat.shape[1] // block_size
    return [cat[:, i * block_size : (i + 1) * block_size] for i in range(n_split)]
