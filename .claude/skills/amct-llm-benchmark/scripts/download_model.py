"""Download a HuggingFace-format model from ModelScope.

Usage:
    python download_model.py --model_id Qwen/Qwen3-8B --local_dir ./Qwen3-8B
    python download_model.py --model_id Qwen/Qwen3-8B --local_dir ./Qwen3-8B \
        --include "*.safetensors" "*.json" "tokenizer*"

Notes:
- Requires `pip install modelscope`.
- Skips re-download for files that already exist (resumable).
- For very large models, pin --include to skip non-safetensors checkpoints.
"""
import argparse
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True, help="ModelScope model id, e.g. Qwen/Qwen3-8B")
    p.add_argument("--local_dir", required=True, help="Target directory")
    p.add_argument("--include", nargs="*", default=None,
                   help="Glob patterns to include (skip everything else)")
    args = p.parse_args()

    try:
        from modelscope import snapshot_download
    except ImportError:
        print("modelscope is not installed. Run: pip install modelscope", file=sys.stderr)
        return 1

    kwargs = {"local_dir": args.local_dir}
    if args.include:
        kwargs["allow_file_pattern"] = args.include

    path = snapshot_download(args.model_id, **kwargs)
    print(f"[download] saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
