# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
"""Export an AMCT-quantized model to an offline torchair AIR graph.

The input model must already be the deploy model produced by ``amct.convert``.
This script only does the export part:

    load quantized model -> build a real forward input -> torch.compile with
    torchair npu backend -> ``tng.dynamo_export`` -> <name>.air + dynamo.pbtxt
    + input.bin

Usage:
    # LLM (e.g. Qwen/Llama), model saved via torch.save after amct.convert:
    python3 src/export_torchair_sample.py --model_path /path/to/quantized_llm \
        --seqlen 2048

    # VLM (CLIP-ViT like):
    python3 src/export_torchair_sample.py --model_type vlm \
        --model_path /path/to/quantized_clip --image_path /path/to/img.jpg

Note: the quantized deploy model must be saved with ``torch.save(quant_model,
<path>/model.pth)`` right after ``amct.convert`` (see README), because the
deploy modules (e.g. NpuQuantizationLinear) are AMCT custom modules that
``AutoModelForCausalLM.from_pretrained`` cannot restore from safetensors.
"""

import argparse
import os

# NPU / torchair env knobs (users may override them before running)
os.environ.setdefault("ACL_OP_COMPILER_CACHE_MODE", "disable")
os.environ.setdefault("ENABLE_DYNAMIC_SHAPE", "false")
os.environ.setdefault("ENABLE_ACLNN", "false")

import torch  # noqa: E402
import torch_npu  # noqa: E402, F401
import torchair as tng  # noqa: E402
from torchair.configs.compiler_config import CompilerConfig  # noqa: E402

from transformers import AutoProcessor, AutoTokenizer  # noqa: E402

# Fallback text used to build the LLM export input when --input_file is not
# given; only its token length matters, not its content.
DEFAULT_INPUT_TEXT = (
    "The quick brown fox jumps over the lazy dog. " * 64
)


def load_llm(model_path):
    """Load an AMCT-quantized causal LM saved by torch.save (model.pth).

    The deploy model contains AMCT custom modules (NpuQuantizationLinear),
    which can only be restored via torch.load. Modify me if needed.
    """
    pth_path = model_path
    if os.path.isdir(model_path):
        pth_path = os.path.join(model_path, "model.pth")
    print(f"Loading quantized LLM from {pth_path}")
    model = torch.load(pth_path, map_location="cpu", weights_only=False)
    # eager attention avoids sdpa, which torch_npu redispatches to
    # npu_fusion_attention_v3 (no AscendIR converter in torchair yet)
    model.config._attn_implementation = "eager"
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=False,
                                        trust_remote_code=True)
    return model, enc


def load_vlm(model_path):
    """Load an AMCT-quantized CLIP-ViT model saved by torch.save (model.pth).

    Modify me if needed.
    """
    pth_path = model_path
    if os.path.isdir(model_path):
        pth_path = os.path.join(model_path, "model.pth")
    print(f"Loading quantized VLM from {pth_path}")
    model = torch.load(pth_path, map_location="cpu", weights_only=False)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def to_npu(model):
    """Move the model to NPU, fixing tensors that AMCT deploy modules keep as
    plain attributes (e.g. NpuQuantizationLinear.bias/offset_bias) and are
    therefore not moved by Module.npu()."""
    model = model.npu()
    for mod in model.modules():
        for attr in ("bias", "offset_bias"):
            value = getattr(mod, attr, None)
            if torch.is_tensor(value) and value.device.type == "cpu":
                setattr(mod, attr, value.npu())
    return model


def build_llm_input(enc, input_file, seqlen):
    """Tokenize real text into the export input [1, seqlen] int64."""
    if input_file is not None:
        with open(input_file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = DEFAULT_INPUT_TEXT
    token_ids = enc(text, return_tensors="pt").input_ids
    if token_ids.shape[1] < seqlen:
        # repeat the text until it covers seqlen tokens
        repeats = seqlen // token_ids.shape[1] + 1
        token_ids = token_ids.repeat(1, repeats)
    export_input = token_ids[:, :seqlen].npu()
    print(f"LLM export input shape: {tuple(export_input.shape)}, "
          f"dtype: {export_input.dtype}")
    return export_input


def build_vlm_input(processor, image_path, batch_size, dtype):
    """Preprocess one real image into the export input [bs, 3, H, W]."""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    export_input = pixel_values.repeat(batch_size, 1, 1, 1).to(dtype).npu()
    print(f"VLM export input shape: {tuple(export_input.shape)}, "
          f"dtype: {export_input.dtype}")
    return export_input


def export_torchair_air(model, export_input, output_dir, export_name):
    """torch.compile with the torchair npu backend, then export the AIR graph."""
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    torch.npu.config.allow_internal_format = True
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    config.export.experimental.enable_lite_export = True

    os.makedirs(output_dir, exist_ok=True)
    input_bin = os.path.join(output_dir, "input.bin")
    export_input.cpu().numpy().tofile(input_bin)
    print(f"input.bin written ({os.path.getsize(input_bin)} B)")

    print(f"=== Exporting {export_name} to {output_dir} ===")
    tng.dynamo_export(
        export_input,
        model=model,
        export_path=output_dir,
        export_name=export_name,
        dynamic=False,
        config=config,
    )
    print(f"=== Export DONE: ['{export_name}.air', 'dynamo.pbtxt', 'input.bin'] ===")


def main():
    parser = argparse.ArgumentParser(
        description="Export an AMCT-quantized model to an offline torchair AIR graph"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path of the AMCT-quantized deploy model "
                             "(already converted by amct.convert)")
    parser.add_argument("--model_type", type=str, choices=["llm", "vlm"],
                        default="llm", help="llm: causal LM; vlm: CLIP-ViT like")
    parser.add_argument("--output_dir", type=str, default="./torchair_output",
                        help="Directory for exported artifacts")
    parser.add_argument("--export_name", type=str, default=None,
                        help="AIR file name without suffix "
                             "(default: basename of model_path)")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Sequence length of the LLM export input")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Text file tokenized as the LLM export input "
                             "(default: built-in sample text)")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Real image used to trace the VLM forward (vlm only)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size of the VLM export input")
    parser.add_argument("--dtype", type=str,
                        choices=["float16", "bfloat16"], default="float16",
                        help="Dtype of the VLM image input")
    args = parser.parse_args()

    if args.model_type == "vlm" and not args.image_path:
        parser.error("--image_path is required when --model_type is vlm")

    export_name = args.export_name or os.path.basename(
        args.model_path.rstrip("/"))

    if args.model_type == "llm":
        model, enc = load_llm(args.model_path)
        model = to_npu(model.eval())
        export_input = build_llm_input(enc, args.input_file, args.seqlen)
    else:
        dtype = getattr(torch, args.dtype)
        model, processor = load_vlm(args.model_path)
        model = to_npu(model.eval())
        export_input = build_vlm_input(processor, args.image_path,
                                       args.batch_size, dtype)

    export_torchair_air(model, export_input, args.output_dir, export_name)


if __name__ == "__main__":
    main()
