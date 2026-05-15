# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Apache License, 2.0
"""
HiFloat8 Cast 测试统一入口

用法：
    python scripts/run_tests.py                  # 运行全部测试
    python scripts/run_tests.py --test accuracy  # 仅运行精度测试
    python scripts/run_tests.py --test perf      # 仅运行性能测试
    python scripts/run_tests.py --test cpu-npu   # 运行 CPU vs NPU 对比测试（精度 + 性能）
"""

import argparse
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script_name, extra_args=None):
    """运行测试脚本"""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    cmd = [sys.executable, script_path]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n运行: {script_name}")
    print("=" * 70)
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="HiFloat8 Cast 测试入口")
    parser.add_argument('--test', type=str, choices=['accuracy', 'perf', 'cpu-npu', 'all'],
                        default='all', help='测试类型')
    args = parser.parse_args()

    results = {}

    if args.test in ['accuracy', 'all']:
        results['accuracy'] = run_script('test_accuracy.py')

    if args.test in ['perf', 'all']:
        results['performance'] = run_script('test_performance.py')

    if args.test in ['cpu-npu', 'all']:
        print("\n注意: CPU vs NPU 测试需要额外依赖")
        print("      需先构建 amct_pytorch/experimental/hifloat8/hifloat8_cast.so")
        results['accuracy_cpu_npu'] = run_script('test_accuracy_cpu_npu.py')
        results['performance_cpu_npu'] = run_script('test_performance_cpu_npu.py')

    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    for name, ret in results.items():
        status = "PASS" if ret == 0 else "FAIL"
        print(f"  {name:20s}: {status}")

    all_pass = all(ret == 0 for ret in results.values())
    print("\n" + ("全部测试通过 ✓" if all_pass else "部分测试失败"))
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())