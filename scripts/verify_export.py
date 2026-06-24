#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""量化导出产物校验工具（与具体模型解耦）。

校验 amct deploy 导出目录的 int 量化产物是否“反量化保真”：
对每个量化 Linear，比较
  - 导出侧反量化:  w_deq = weight(int8).float() * weight_scale
  - 参考侧伪量化:  对原始 BF16 权重按同 scale 做 round->clamp->dequant
两者应逐元素一致（导出 = 量化方案的忠实落盘），并报告反量化权重 vs 原始 BF16 的
余弦相似度 / 最大相对误差（量化引入的权重误差量级）。

只读校验，不跑模型前向、不需要完整模型；按需抽样层。

用法:
  python verify_export.py --deploy_dir <导出目录> [--ref_model <原始BF16目录>]
                          [--layers 0,1,2] [--max_tensors_per_layer 8]
"""
import argparse
import json
import os
import struct

import torch


def read_safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data_start = 8 + n
    return header, data_start


_DTYPE = {
    "I8": torch.int8, "U8": torch.uint8, "F16": torch.float16,
    "BF16": torch.bfloat16, "F32": torch.float32, "I32": torch.int32,
    "I64": torch.int64,
}


def load_tensor(path, header, data_start, key):
    meta = header[key]
    dt = _DTYPE[meta["dtype"]]
    s, e = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + s)
        buf = f.read(e - s)
    t = torch.frombuffer(bytearray(buf), dtype=dt)
    return t.reshape(meta["shape"])


def find_layer_files(deploy_dir, layers):
    out = {}
    for fn in os.listdir(deploy_dir):
        if fn.startswith("layer_") and fn.endswith(".safetensors"):
            idx = int(fn[len("layer_"):-len(".safetensors")])
            if layers is None or idx in layers:
                out[idx] = os.path.join(deploy_dir, fn)
    return dict(sorted(out.items()))


def dequant_int8_perchannel(w_int8, scale):
    # scale: [out,1]；w: [out,in]
    return w_int8.to(torch.float32) * scale.to(torch.float32)


def quant_dequant_ref(w_bf16, scale, qmin=-128, qmax=127):
    # 用同一 scale 把原始权重伪量化->反量化（int 对称 per-channel）
    w = w_bf16.to(torch.float32)
    q = torch.round(w / scale.to(torch.float32)).clamp(qmin, qmax)
    return q * scale.to(torch.float32)


def cosine(a, b):
    # 裁剪到 [-1,1]，规避浮点解析带来的微小越界
    c = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    return max(-1.0, min(1.0, c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy_dir", required=True)
    ap.add_argument("--ref_model", default=None, help="原始 BF16 模型目录（可选，做量化误差对比）")
    ap.add_argument("--layers", default=None, help="逗号分隔层号，默认全部导出层")
    ap.add_argument("--max_tensors_per_layer", type=int, default=8)
    args = ap.parse_args()

    layers = None
    if args.layers:
        layers = set(int(x) for x in args.layers.split(","))
    layer_files = find_layer_files(args.deploy_dir, layers)
    print(f"[verify] deploy_dir={args.deploy_dir}")
    print(f"[verify] layers to check: {list(layer_files.keys())}")

    # 参考模型 weight_map（可选）
    ref_index = None
    if args.ref_model:
        idx_path = os.path.join(args.ref_model, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            ref_index = json.load(open(idx_path))["weight_map"]

    total_q, self_consistent, cos_sum, maxrel_max = 0, 0, 0.0, 0.0
    ref_checked = 0
    issues = []

    for idx, path in layer_files.items():
        header, ds = read_safetensors_header(path)
        scale_keys = [k for k in header if k.endswith(".weight_scale")]
        checked = 0
        for sk in scale_keys:
            if checked >= args.max_tensors_per_layer:
                break
            wk = sk[:-len("_scale")]  # *.weight
            if wk not in header:
                issues.append(f"L{idx}: {sk} 无配对 {wk}")
                continue
            if header[wk]["dtype"] != "I8":
                issues.append(f"L{idx}: {wk} dtype={header[wk]['dtype']} 非 I8")
                continue
            w_int8 = load_tensor(path, header, ds, wk)
            scale = load_tensor(path, header, ds, sk)
            total_q += 1
            checked += 1

            # 自洽性：导出反量化值落在 int8*scale 网格上（必然，但验 scale 维度/形状对齐）
            w_deq = dequant_int8_perchannel(w_int8, scale)
            if w_deq.shape != tuple(header[wk]["shape"]):
                issues.append(f"L{idx}: {wk} 反量化形状不符")
            else:
                self_consistent += 1

            # 与原始 BF16 对比（若给了 ref_model）
            if ref_index is not None and wk in ref_index:
                shard = os.path.join(args.ref_model, ref_index[wk])
                if os.path.exists(shard):
                    rh, rds = read_safetensors_header(shard)
                    if wk in rh:
                        w_ref = load_tensor(shard, rh, rds, wk).to(torch.float32)
                        ref_deq = quant_dequant_ref(w_ref, scale)
                        # 导出反量化 应等于 对原始权重同 scale 伪量化（验证导出忠实）。
                        # 容差按量化步长：边界元素 round 方向差异最多偏 1 个 scale 单位。
                        tol = scale.to(torch.float32).abs().max().item() * 1.001
                        match = torch.allclose(w_deq, ref_deq, atol=tol)
                        cos = cosine(w_deq, w_ref)
                        denom = w_ref.abs().mean().clamp_min(1e-8)
                        maxrel = ((w_deq - w_ref).abs().mean() / denom).item()
                        cos_sum += cos
                        maxrel_max = max(maxrel_max, maxrel)
                        ref_checked += 1
                        if not match:
                            issues.append(f"L{idx}: {wk} 导出反量化≠原始同scale伪量化(可能 scale 口径不一致)")

    print(f"\n[verify] 量化张量数: {total_q}; 形状自洽: {self_consistent}/{total_q}")
    if ref_checked:
        print(f"[verify] vs 原始BF16: 抽查 {ref_checked} 个; "
              f"平均余弦 {cos_sum/ref_checked:.6f}; 最大相对误差 {maxrel_max:.4f}")
    else:
        print("[verify] 未做原始对比（未给 --ref_model 或 index 缺失）")
    if issues:
        print(f"\n[verify] 发现 {len(issues)} 个问题:")
        for s in issues[:40]:
            print("  -", s)
    else:
        print("\n[verify] 通过：所有抽查量化张量 weight/scale 配对、I8、形状自洽" +
              ("，且反量化与原始 BF16 同 scale 伪量化一致" if ref_checked else ""))


if __name__ == "__main__":
    main()
