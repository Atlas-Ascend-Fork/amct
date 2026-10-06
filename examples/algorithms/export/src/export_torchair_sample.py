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

    # VLM (CLIP-ViT like, image-only input), image_path can be an image file
    # or a directory containing one/multiple images:
    python3 src/export_torchair_sample.py --model_type vlm \
        --model_path /path/to/quantized_clip --image_path /path/to/img.jpg

    python3 src/export_torchair_sample.py --model_type vlm \
        --model_path /path/to/quantized_clip --image_path /path/to/img_dir

Note: this script only supports image-only VLM models (e.g. CLIP); text+image
multimodal VLMs (Qwen2-VL / Qwen2.5-VL / LLaVA ...) are not supported.

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
DEFAULT_INPUT_TEXT = "The quick brown fox jumps over the lazy dog. " * 64


def load_llm(model_path, trust_remote_code=False):
    """Load an AMCT-quantized causal LM saved by torch.save (model.pth).

    The deploy model contains AMCT custom modules (NpuQuantizationLinear),
    which can only be restored via torch.load. Modify me if needed.
    """
    pth_path = model_path
    if os.path.isdir(model_path):
        pth_path = os.path.join(model_path, "model.pth")
        tokenizer_path = model_path
    else:
        tokenizer_path = os.path.dirname(os.path.abspath(model_path))
    print(f"Loading quantized LLM from {pth_path}")
    model = torch.load(pth_path, map_location="cpu", weights_only=False)
    # eager attention avoids sdpa, which torch_npu redispatches to
    # npu_fusion_attention_v3 (no AscendIR converter in torchair yet)
    model.config._attn_implementation = "eager"
    # disable KV cache: some transformers versions initialize DynamicCache
    # with an empty tensor whose torch.cat([empty, key_states], dim=-2) is
    # traced as a 0-dim ConcatV2 that the GE InferShape cannot handle
    model.config.use_cache = False
    enc = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False, trust_remote_code=trust_remote_code
    )
    return model, enc


def load_vlm(model_path, trust_remote_code=False):
    """Load an AMCT-quantized image-only VLM saved by torch.save (model.pth).

    The pickled whole model preserves its original class (CLIPModel, ...), so
    no class-specific loading is needed. Modify me if needed.
    """
    pth_path = model_path
    if os.path.isdir(model_path):
        pth_path = os.path.join(model_path, "model.pth")
        processor_path = model_path
    else:
        processor_path = os.path.dirname(os.path.abspath(model_path))
    print(f"Loading quantized VLM from {pth_path}")
    model = torch.load(pth_path, map_location="cpu", weights_only=False)
    # eager attention avoids sdpa, which torch_npu redispatches to
    # npu_fusion_attention_v3 (no AscendIR converter in torchair yet);
    # applies to CLIP vision tower alike
    for cfg_name in ("config", "text_config", "vision_config"):
        cfg = getattr(model, cfg_name, None)
        if cfg is not None:
            cfg._attn_implementation = "eager"
    processor = AutoProcessor.from_pretrained(
        processor_path, trust_remote_code=trust_remote_code
    )
    return model, processor


class CLIPImageFeaturesModule(torch.nn.Module):
    """Wrap ``CLIPModel.get_image_features`` as a callable nn.Module so that
    ``dynamo_export`` traces only the vision tower with a single
    ``pixel_values`` input (CLIPModel.forward itself requires both
    ``input_ids`` and ``pixel_values`` and is not what downstream image
    embedding inference wants)."""

    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, pixel_values):
        # Inline ``CLIPModel.get_image_features`` (vision tower -> pooled [CLS]
        # -> visual projection) so the traced output is a plain tensor that is
        # stable across transformers versions (>= 4.x may return a
        # BaseModelOutputWithPooling instead of a tensor).
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        return self.clip.visual_projection(pooled_output)


def to_npu(model):
    """Move the model to NPU, fixing tensors that AMCT deploy modules keep as
    plain attributes (e.g. NpuQuantizationLinear.bias/offset_bias/y_scale) and are
    therefore not moved by Module.npu()."""
    model = model.npu()
    for mod in model.modules():
        for attr in ("bias", "offset_bias", "y_scale"):
            value = getattr(mod, attr, None)
            if torch.is_tensor(value) and value.device.type == "cpu":
                setattr(mod, attr, value.npu())
    return model


def build_llm_input(enc, input_file, seqlen):
    """Tokenize real text into the export input [1, seqlen] int64."""
    if seqlen <= 0:
        raise ValueError(
            f"--seqlen 必须为正整数，当前值为 {seqlen}，请检查 --seqlen 参数"
        )
    if input_file is not None:
        with open(input_file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = DEFAULT_INPUT_TEXT
    token_ids = enc(text, return_tensors="pt").input_ids
    if token_ids.shape[1] == 0:
        raise ValueError(
            "--input_file 指向的文件 token 数为 0（可能为空文件），"
            "请检查 --input_file 指向的文件内容"
        )
    if token_ids.shape[1] < seqlen:
        # repeat the text until it covers seqlen tokens
        repeats = seqlen // token_ids.shape[1] + 1
        token_ids = token_ids.repeat(1, repeats)
    export_input = token_ids[:, :seqlen].npu()
    print(
        f"LLM export input shape: {tuple(export_input.shape)}, "
        f"dtype: {export_input.dtype}"
    )
    return (export_input,)


def _detect_vlm_kind(model):
    """Verify the model is an image-only VLM (CLIP-like).

    Returns 'clip' for CLIPModel (image-only forward). Raises a clear error
    for text+image multimodal VLMs (Qwen2-VL / Qwen2.5-VL / LLaVA ...).
    """
    cls_name = type(model).__name__.lower()
    if "clip" in cls_name and "vision" not in cls_name:
        return "clip"
    model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
    if "clip" in model_type and "vision" not in model_type:
        return "clip"
    raise NotImplementedError(
        "当前脚本仅支持纯图输入的VLM模型（如CLIP/CLIPModel）。"
        f"检测到不支持的模型类型: {type(model).__name__ or model_type or 'unknown'}，"
        "图文混合输入的多模态VLM（Qwen2-VL/LLaVA等）暂不支持。"
    )


def _load_images(image_path, max_count=None):
    """Load at most ``max_count`` images from a file or directory.

    For a directory, image files are listed first (without decoding) so the
    count can be validated, and only the first ``max_count`` images are
    decoded to avoid loading unnecessary images into memory.

    Returns a list of PIL Image objects.
    """
    from PIL import Image

    if os.path.isfile(image_path):
        if max_count is not None and max_count > 1:
            raise ValueError(
                f"batch_size ({max_count}) 大于图片数量 (1)，"
                "请提供足够多的图片或减小 batch_size"
            )
        return [Image.open(image_path).convert("RGB")]
    elif os.path.isdir(image_path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        filenames = sorted(
            f
            for f in os.listdir(image_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        )
        if not filenames:
            raise ValueError(f"目录中未找到图片文件: {image_path}")
        if max_count is not None and max_count > len(filenames):
            raise ValueError(
                f"batch_size ({max_count}) 大于图片数量 ({len(filenames)})，"
                "请提供足够多的图片或减小 batch_size"
            )
        images = [
            Image.open(os.path.join(image_path, f)).convert("RGB")
            for f in filenames[:max_count]
        ]
        print(f"从目录 {image_path} 解码了 {len(images)} 张图片")
        return images
    else:
        raise ValueError(f"image_path 必须是文件或目录: {image_path}")


def _default_vlm_input_size(processor):
    """从 processor 推断默认输入尺寸，供未提供图片时的占位输入使用。"""
    size = getattr(processor, "crop_size", None) or getattr(processor, "size", None)
    if isinstance(size, dict):
        size = size.get("height") or size.get("shortest_edge") or size.get("width")
    if size is None:
        size = 224
    if isinstance(size, (tuple, list)):
        size = size[0]
    return int(size)


def _validate_model_dtype(model, input_dtype):
    """检查模型浮点参数 dtype 与输入 --dtype 是否一致。

    仅校验浮点参数（跳过 int8/int64 等量化权重）；不一致时给出指向
    --dtype 的报错，避免导出时算子报 dtype 不匹配。
    """
    for param in model.parameters():
        if param.dtype in (torch.float16, torch.bfloat16, torch.float32):
            if param.dtype != input_dtype:
                raise ValueError(
                    f"--dtype 指定的输入精度为 {input_dtype}，但模型浮点权重为 "
                    f"{param.dtype}，两者不一致会导致算子报 dtype 不匹配错误；"
                    "请将 --dtype 调整为与模型权重一致的精度，或先将模型转为目标精度"
                )
            return


def build_vlm_input(model, processor, image_path, batch_size, dtype, device):
    """Build the VLM export inputs by preprocessing images.

    Only image-only VLMs (CLIP-like) are supported. ``image_path`` may be a
    single image file, a directory containing one/multiple images, or None.
    When it is None, a placeholder ``pixel_values`` with the processor default
    size is used so the export can still trace the vision tower. The model is
    wrapped in ``CLIPImageFeaturesModule`` so only the vision tower is
    traced.

    Returns (model, tuple_of_tensors); the model is replaced by the wrapper.
    """
    if batch_size <= 0:
        raise ValueError(
            f"--batch_size 必须为正整数，当前值为 {batch_size}，请检查 --batch_size 参数"
        )
    _validate_model_dtype(model, dtype)
    _detect_vlm_kind(model)
    model = CLIPImageFeaturesModule(model)

    if image_path is None:
        size = _default_vlm_input_size(processor)
        pixel_values = torch.randn(
            batch_size, 3, size, size, dtype=dtype, device=device
        )
    else:
        images = _load_images(image_path, max_count=batch_size)
        pixel_values_list = []
        for img in images:
            inputs = processor(images=img, return_tensors="pt")
            pixel_values_list.append(inputs["pixel_values"])

        if len(pixel_values_list) == 1:
            pixel_values = pixel_values_list[0]
        else:
            pixel_values = torch.cat(pixel_values_list, dim=0)

    export_input = (pixel_values.to(dtype).to(device),)
    print(
        f"VLM(clip image tower) export input: "
        f"{[(tuple(t.shape), str(t.dtype)) for t in export_input]}"
    )
    return model, export_input


def _patch_weight_subdir():
    """Make torchair dump externalized weights into ``<export_dir>/weights``
    instead of scattered alongside the ``.air`` file.

    torchair writes externalized weight files (and ``dynamo.pbtxt``) to the
    path returned by ``_get_subpath``. The stock implementation returns the
    export dir itself (only adding ``rank_N`` under distributed mode), so large
    models (e.g. >2G protobuf) scatter hundreds of weight files next to the
    ``.air``. We patch it to append a ``weights`` subdirectory so the export
    dir stays clean and ``.air`` / ``input.bin`` stand alone.
    """
    import importlib

    targets = (
        "torchair._utils.export_utils",
        "torch_npu.dynamo.torchair._utils.export_utils",
    )
    for mod_name in targets:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue
        if getattr(mod, "_weight_subdir_patched", False):
            return
        orig_get_subpath = mod._get_subpath

        def patched(export_path_dir, _orig=orig_get_subpath):
            return _orig(export_path_dir) + "/weights"

        mod._get_subpath = patched
        mod._weight_subdir_patched = True
        print(
            f"weight files will be dumped to <export_dir>/weights (patched {mod_name})"
        )
        return


def export_torchair_air(model, export_inputs, output_dir, export_name):
    """torch.compile with the torchair npu backend, then export the AIR graph.

    ``export_inputs`` is a sequence of tensors splatted into
    ``tng.dynamo_export`` (LLM: 1 tensor; image-only VLM: 1 pixel_values
    tensor).

    ``torch.compile`` wraps the model as ``OptimizedModule`` whose
    ``named_buffers`` carry an ``_orig_mod.`` prefix. torchair uses those names
    as exported weight file names, producing ``_orig_mod_xxxx_weight`` files.
    Unwrap to the original module before compile so the names stay clean.
    """
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    _patch_weight_subdir()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    torch.npu.config.allow_internal_format = True
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    config.export.experimental.enable_lite_export = True

    os.makedirs(output_dir, exist_ok=True)
    for i, tensor in enumerate(export_inputs):
        suffix = f"_{i}" if len(export_inputs) > 1 else ""
        input_bin = os.path.join(output_dir, f"input{suffix}.bin")
        if tensor.dtype == torch.bfloat16:
            # bfloat16 无法直接转 numpy（numpy 无对应类型），按 uint16
            # 保持原始位模式落盘，避免出现不支持 ScalarType 的报错
            tensor.cpu().contiguous().view(torch.uint16).numpy().tofile(input_bin)
        else:
            tensor.cpu().numpy().tofile(input_bin)
        print(f"input{suffix}.bin written ({os.path.getsize(input_bin)} B)")

    print(f"=== Exporting {export_name} to {output_dir} ===")
    tng.dynamo_export(
        *export_inputs,
        model=model,
        export_path=output_dir,
        export_name=export_name,
        dynamic=False,
        config=config,
    )
    artifacts = [f"{export_name}.air"]
    if len(export_inputs) > 1:
        artifacts.append("input_<i>.bin")
    elif len(export_inputs) == 1:
        artifacts.append("input.bin")
    if os.path.isdir(os.path.join(output_dir, "weights")):
        artifacts.append("weights/dynamo.pbtxt + weight files")
    else:
        artifacts.append("dynamo.pbtxt")
    print(f"=== Export DONE: {artifacts} ===")


def main():
    parser = argparse.ArgumentParser(
        description="Export an AMCT-quantized model to an offline torchair AIR graph"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path of the quantized deploy model or its "
        "containing directory (already converted by "
        "amct.convert)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["llm", "vlm"],
        default="llm",
        help="llm: causal LM; vlm: image-only VLM (e.g. CLIP)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./torchair_output",
        help="Directory for exported artifacts",
    )
    parser.add_argument(
        "--export_name",
        type=str,
        default=None,
        help="AIR file name without suffix (default: model directory name for a "
        "directory, or file name without suffix for a .pth file)",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="Sequence length of the LLM export input",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Text file tokenized as the LLM export input "
        "(default: built-in sample text)",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Image file or a directory of images used to "
        "trace the VLM forward (vlm only, optional; "
        "if not provided, a placeholder input with the "
        "processor default size is used)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size of the VLM export input; must be <= "
        "the number of images under --image_path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Dtype of the VLM image input; only applies to "
        "the vlm scenario, because the LLM export input "
        "is int64 token ids and has no float precision "
        "to tune",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Execute custom Python code in the model directory "
        "(disabled by default; only enable for trusted local models)",
    )
    args = parser.parse_args()

    if args.trust_remote_code:
        print(
            "WARNING: --trust_remote_code is enabled. Custom Python from the "
            "model directory may execute under the current user. Use only "
            "with checkpoints you trust.",
            flush=True,
        )

    if args.model_type == "llm" and args.image_path is not None:
        print(
            f"Warning: --image_path ({args.image_path}) 仅适用于vlm场景，"
            "llm场景下该参数未生效。"
        )
    if args.model_type == "llm" and args.batch_size != 1:
        print(
            f"Warning: --batch_size ({args.batch_size}) 仅适用于vlm场景，"
            "llm场景下该参数未生效。"
        )
    if args.model_type == "vlm" and args.seqlen != 2048:
        print(
            f"Warning: --seqlen ({args.seqlen}) 仅适用于llm场景，"
            "vlm场景下该参数未生效。"
        )
    if args.model_type == "vlm" and args.input_file is not None:
        print(
            f"Warning: --input_file ({args.input_file}) 仅适用于llm场景，"
            "vlm场景下该参数未生效。"
        )
    if (
        args.model_type == "vlm"
        and args.image_path is not None
        and not os.path.exists(args.image_path)
    ):
        parser.error(f"--image_path does not exist: {args.image_path}")

    export_name = args.export_name or os.path.basename(args.model_path.rstrip("/"))
    if args.export_name is None and os.path.isfile(args.model_path):
        # model_path 指向文件时去掉 .pth 后缀，避免导出名变成 model.pth.air
        export_name = os.path.splitext(export_name)[0]
    # 将 output_dir 归一化为绝对路径，避免 air 中 FileConstant 外置权重
    # 使用相对路径，导致 ATC 转 om 时因基准目录不一致而找不到权重
    args.output_dir = os.path.abspath(args.output_dir)

    if args.model_type == "llm":
        model, enc = load_llm(args.model_path, args.trust_remote_code)
        model = to_npu(model.eval())
        export_inputs = build_llm_input(enc, args.input_file, args.seqlen)
    else:
        model, processor = load_vlm(args.model_path, args.trust_remote_code)
        model = to_npu(model.eval())
        dtype = getattr(torch, args.dtype)
        device = next(model.parameters()).device
        model, export_inputs = build_vlm_input(
            model, processor, args.image_path, args.batch_size, dtype, device
        )

    export_torchair_air(model, export_inputs, args.output_dir, export_name)


if __name__ == "__main__":
    main()
