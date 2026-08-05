"""
将 configs/ 中的 COMMON_DATA_PATH 占位符替换为实际数据集路径，
输出到 configs_mod/，同时修补 datasets/ 中各 .py 脚本的路径。

用法：
    python set_path.py <data_path>
        data_path: 数据集根目录的绝对路径（包含 ARC-C、hellaswag 等子目录）

也可通过环境变量指定：
    EVAL_DATA_PATH=/path/to/datasets python set_path.py
"""

import os
import sys
import shutil
import yaml


# ── 解析 data_path ─────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    new_common_path = sys.argv[1].rstrip('/')
elif 'EVAL_DATA_PATH' in os.environ:
    new_common_path = os.environ['EVAL_DATA_PATH'].rstrip('/')
else:
    print(__doc__)
    print("错误：请提供数据集路径。")
    sys.exit(1)

PLACEHOLDER = "COMMON_DATA_PATH"
eval_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.join(eval_dir, "configs")
dst_root = os.path.join(eval_dir, "configs_mod")
datasets_root = os.path.join(eval_dir, "datasets")


# ── 1. 修补 datasets/ 下的 .py 脚本 ───────────────────────────────────────
class _LMEvalLoader(yaml.SafeLoader):
    pass

yaml.add_constructor(
    '!function',
    lambda loader, node: loader.construct_scalar(node),
    _LMEvalLoader,
)

patched_py = 0
for root, dirs, files in os.walk(datasets_root):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    for f in files:
        if f.endswith('.py'):
            fpath = os.path.join(root, f)
            with open(fpath) as fh:
                content = fh.read()
            if PLACEHOLDER in content:
                new_content = content.replace(PLACEHOLDER, new_common_path)
                with open(fpath, 'w') as fh:
                    fh.write(new_content)
                rel = os.path.relpath(fpath, eval_dir)
                print(f"[datasets] patched: {rel}")
                patched_py += 1


# ── 2. 复制 configs/ → configs_mod/，替换 COMMON_DATA_PATH ────────────────
if os.path.exists(dst_root):
    shutil.rmtree(dst_root)

copied = 0
for root, dirs, files in os.walk(src_root):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    rel_dir = os.path.relpath(root, src_root)
    dst_dir = os.path.join(dst_root, rel_dir)
    os.makedirs(dst_dir, exist_ok=True)

    for f in files:
        src_file = os.path.join(root, f)
        dst_file = os.path.join(dst_dir, f)

        if f.endswith('.yaml') or not os.path.splitext(f)[1]:
            with open(src_file) as fh:
                content = fh.read()
            if PLACEHOLDER in content:
                content = content.replace(PLACEHOLDER, new_common_path)
                rel = os.path.relpath(src_file, src_root)
                print(f"[configs]  replaced: {rel}")
            with open(dst_file, 'w') as fh:
                fh.write(content)
        else:
            shutil.copy2(src_file, dst_file)
        copied += 1

print(f"\n完成：修补 {patched_py} 个 .py 脚本，生成 {copied} 个配置文件 → {dst_root}")
