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

"""Rewrite one safetensors shard in place of load_file plus save_file.

save_file needs every tensor of a shard resident at once. DeepSeek V4.1 ships
two 94.6 GiB shards whose engram table dwarfs the 0.15 GiB a conversion touches,
so tensors that are not replaced are copied byte range to byte range here and
never become torch objects.
"""

import json
import os
import struct

import torch

COPY_CHUNK = 64 << 20

DTYPE_STR = {
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.bool: "BOOL",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e5m2: "F8_E5M2",
    torch.float8_e8m0fnu: "F8_E8M0",
}


def rewrite_shard(source_path, destination_path, replacements):
    """Copy a shard to destination_path, swapping in replacements by name.

    Returns the tensor names the destination holds, in the source's order.
    """
    with open(source_path, "rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_length))
    data_start = 8 + header_length

    new_header, cursor, plan = {}, 0, []
    for name in [key for key in header if key != "__metadata__"]:
        if name in replacements:
            tensor = replacements[name].contiguous()
            size = tensor.numel() * tensor.element_size()
            if tensor.dtype not in DTYPE_STR:
                raise ValueError(f"{name}: unsupported dtype {tensor.dtype}")
            entry = {"dtype": DTYPE_STR[tensor.dtype], "shape": list(tensor.shape)}
            plan.append(("new", name, size))
        else:
            source_entry = header[name]
            low, high = source_entry["data_offsets"]
            size = high - low
            entry = {"dtype": source_entry["dtype"], "shape": source_entry["shape"]}
            plan.append(("copy", data_start + low, size))
        entry["data_offsets"] = [cursor, cursor + size]
        new_header[name] = entry
        cursor += size
    new_header["__metadata__"] = {"format": "pt"}

    blob = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)  # keep the data block 8-byte aligned
    temporary = f"{destination_path}.tmp"
    with open(source_path, "rb") as source, open(temporary, "wb") as destination:
        destination.write(struct.pack("<Q", len(blob)))
        destination.write(blob)
        for kind, locator, size in plan:
            if kind == "new":
                tensor = replacements[locator].contiguous()
                if tensor.dtype != torch.uint8:
                    tensor = tensor.view(torch.uint8)
                destination.write(tensor.reshape(-1).numpy().tobytes())
                continue
            source.seek(locator)
            left = size
            while left:
                chunk = source.read(min(COPY_CHUNK, left))
                if not chunk:
                    raise IOError(f"{source_path}: short read {left} bytes")
                destination.write(chunk)
                left -= len(chunk)
    os.replace(temporary, destination_path)
    return [key for key in header if key != "__metadata__"]
