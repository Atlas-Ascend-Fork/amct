"""
Stage 1 via vLLM: record SmoothQuant activation statistics on the real vLLM
execution path.

This is intentionally separate from the HuggingFace-based stage1 script. The
goal here is not Quark replay alignment, but to capture activations from the
same vLLM runtime path that is already validated on NPU.

Output filenames are kept stage2-compatible:
- layers_<i>_self_attn_q_proj.pt
- layers_<i>_self_attn_k_proj.pt
- layers_<i>_self_attn_v_proj.pt
- layers_<i>_self_attn_o_proj.pt

Notes:
- q/k/v are recorded from the normalized attention input, using
  `input_layernorm` output and `qkv_proj` input as redundant hook points.
- o_proj is recorded from the attention output, using `attn` output and
  `o_proj` input as redundant hook points before concatenating across TP ranks.
- When tensor parallel > 1, launch this script with `torchrun` and pass the
  same `--tensor_parallel_size`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
import torch.distributed as dist
from tqdm import tqdm

from common import SCALING_LAYERS


DEFAULT_VLLM_REPO_DIR = "/vllm-workspace/vllm"
DEFAULT_VLLM_PATCH_PATH = str(
    Path(__file__).resolve().parent.parent
    / "patches"
    / "0001-MiniMax-M2-adapt-Ascend-fp8-loading-and-qk-norm-path.patch"
)
DEFAULT_COMPILATION_CONFIG = '{"cudagraph_mode":"FULL_DECODE_ONLY"}'


def load_calib_prompts(
    path: str,
    tokenizer,
    num_samples: int,
    seq_len: int,
) -> list[dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= num_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "messages" in data:
                samples.append(data)

    prompts: list[dict[str, Any]] = []
    for sample in samples:
        messages = sample.get("messages", [])
        if not messages:
            continue

        if (
            hasattr(tokenizer, "apply_chat_template")
            and tokenizer.chat_template is not None
        ):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            text = "\n".join(msg.get("content", "") for msg in messages)

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=seq_len,
            add_special_tokens=True,
        )
        input_ids = encoded.get("input_ids", [])
        if not input_ids:
            continue
        prompts.append(
            {
                "prompt": text,
                "prompt_token_ids": input_ids,
            }
        )

    return prompts


def run_command(
    command: list[str], cwd: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def prepend_pythonpath(path: str) -> None:
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    paths = [entry for entry in current_pythonpath.split(":") if entry]
    if path not in paths:
        os.environ["PYTHONPATH"] = (
            f"{path}:{current_pythonpath}" if current_pythonpath else path
        )
    if path not in sys.path:
        sys.path.insert(0, path)


def vllm_patch_is_functionally_present(vllm_repo_dir: str) -> bool:
    repo_path = Path(vllm_repo_dir)
    required_snippets = {
        repo_path / "vllm/config/model.py": "Detected fp8 MiniMax-M2 checkpoint on NPU",
        repo_path
        / "vllm/model_executor/layers/mamba/linear_attn.py": "torch.ops.npu.npu_rms_norm",
        repo_path
        / "vllm/model_executor/models/minimax_m2.py": "_dequantize_fp8_block_weight",
    }

    for path, snippet in required_snippets.items():
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return False
        if snippet not in content:
            return False
    return True


def verify_vllm_patch_applied(vllm_repo_dir: str, patch_path: str) -> None:
    repo_path = Path(vllm_repo_dir)
    patch_file = Path(patch_path)
    if not patch_file.exists():
        raise FileNotFoundError(f"vLLM patch file not found: {patch_file}")
    if vllm_patch_is_functionally_present(str(repo_path)):
        return
    patch_check = run_command(
        ["git", "apply", "--reverse", "--check", str(patch_file)],
        cwd=str(repo_path),
    )
    if patch_check.returncode != 0:
        raise RuntimeError(
            "The required MiniMax vLLM patch does not appear to be applied.\n"
            f"vLLM repo: {repo_path}\n"
            f"Patch: {patch_file}\n"
            "The checker accepts either an exact applied patch or the equivalent "
            "MiniMax Ascend functionality already present in the source tree.\n"
            "Please run ascend/run_vllm_stage1.sh, or apply/rebase the patch "
            "manually before invoking this script directly."
        )


def prepare_vllm_runtime(
    vllm_repo_dir: str,
    patch_path: str | None,
    flashcomm1: str,
) -> None:
    repo_path = Path(vllm_repo_dir).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"vLLM repo directory not found: {repo_path}")
    if not (repo_path / "vllm").exists():
        raise FileNotFoundError(
            f"Expected Python package directory missing: {repo_path / 'vllm'}"
        )

    prepend_pythonpath(str(repo_path))
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = flashcomm1

    if patch_path:
        verify_vllm_patch_applied(str(repo_path), patch_path)


def parse_compilation_config(raw_value: str | None) -> dict[str, Any] | None:
    if raw_value is None:
        return None
    stripped = raw_value.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --compilation_config: {raw_value}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--compilation_config must decode to a JSON object.")
    return parsed


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def barrier() -> None:
    if is_dist_initialized():
        dist.barrier()


def release_accelerator_cache() -> None:
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def stat_tensor(tensor: torch.Tensor) -> torch.Tensor:
    hidden_dim = tensor.shape[-1]
    tensor = tensor.reshape(-1, hidden_dim).abs().detach()
    return torch.max(tensor, dim=0)[0].float()


def update_max(
    store: dict[int, torch.Tensor], layer_idx: int, tensor: torch.Tensor
) -> None:
    current_max = stat_tensor(tensor)
    if layer_idx in store:
        store[layer_idx] = torch.maximum(store[layer_idx], current_max)
    else:
        store[layer_idx] = current_max


def get_worker_model(llm) -> torch.nn.Module:
    model_executor = getattr(llm.llm_engine, "model_executor", None)
    driver_worker = getattr(model_executor, "driver_worker", None)
    worker = getattr(driver_worker, "worker", None)
    if worker is None:
        raise RuntimeError(
            "Failed to locate the in-process vLLM worker. "
            "This script requires VLLM_ENABLE_V1_MULTIPROCESSING=0."
        )
    return worker.get_model()


def reduce_max_in_place(tensor: torch.Tensor) -> torch.Tensor:
    if is_dist_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


def gather_and_concat(tensor: torch.Tensor) -> torch.Tensor:
    if not is_dist_initialized():
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def save_recorded_scales(
    output_dir: str,
    num_layers: int,
    qkv_inputs: dict[int, torch.Tensor],
    o_proj_inputs: dict[int, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(output_dir):
        if filename.endswith(".pt"):
            os.remove(os.path.join(output_dir, filename))

    recorded_keys: list[str] = []
    for layer_idx in range(num_layers):
        if layer_idx not in qkv_inputs:
            raise RuntimeError(f"Missing qkv activation for layer {layer_idx}")
        if layer_idx not in o_proj_inputs:
            raise RuntimeError(f"Missing o_proj activation for layer {layer_idx}")

        qkv_scale = qkv_inputs[layer_idx].cpu()
        o_proj_scale = o_proj_inputs[layer_idx].cpu()

        for proj_name in ("q_proj", "k_proj", "v_proj"):
            key = f"layers.{layer_idx}.self_attn.{proj_name}"
            torch.save(
                qkv_scale, os.path.join(output_dir, f"{key.replace('.', '_')}.pt")
            )
            recorded_keys.append(key)

        o_key = f"layers.{layer_idx}.self_attn.o_proj"
        torch.save(
            o_proj_scale, os.path.join(output_dir, f"{o_key.replace('.', '_')}.pt")
        )
        recorded_keys.append(o_key)

    metadata = dict(metadata)
    metadata["recorded_keys"] = recorded_keys
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def _validate_world_size(env_world_size: int, tensor_parallel_size: int) -> None:
    if env_world_size > 1 and env_world_size != tensor_parallel_size:
        raise ValueError(
            f"WORLD_SIZE({env_world_size}) must match "
            f"--tensor_parallel_size({tensor_parallel_size}) when using torchrun."
        )


def _resolve_max_num_batched_tokens(max_num_batched_tokens: int, seq_len: int) -> int:
    effective = max(max_num_batched_tokens, seq_len + 1)
    if effective != max_num_batched_tokens:
        print(
            "[vLLM Stage 1] Bumping max_num_batched_tokens from "
            f"{max_num_batched_tokens} to {effective} "
            f"to cover seq_len={seq_len} plus one decode token."
        )
    return effective


def _load_tokenizer_and_prompts(
    model_dir: str, calib_data_path: str, num_calib_data: int, seq_len: int
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    print(f"[vLLM Stage 1] Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[vLLM Stage 1] Building calibration prompts from {calib_data_path}...")
    prompts = load_calib_prompts(calib_data_path, tokenizer, num_calib_data, seq_len)
    print(f"[vLLM Stage 1] Loaded {len(prompts)} prompts")
    if not prompts:
        raise RuntimeError("No valid calibration prompts were built.")
    return prompts


def _build_vllm_engine(
    *,
    model_dir: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    seq_len: int,
    enforce_eager: bool,
    max_num_seqs: int,
    effective_max_num_batched_tokens: int,
    enable_expert_parallel: bool,
    compilation_config_dict: dict[str, Any] | None,
    executor_backend: str | None,
):
    from vllm import LLM

    print("[vLLM Stage 1] Initializing vLLM engine...")
    return LLM(
        model=model_dir,
        tokenizer=model_dir,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        enforce_eager=enforce_eager,
        max_model_len=seq_len + 1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=effective_max_num_batched_tokens,
        enable_expert_parallel=enable_expert_parallel,
        compilation_config=compilation_config_dict,
        distributed_executor_backend=executor_backend,
    )


def _make_output_hook(
    target_dict: dict[int, torch.Tensor], layer_idx: int, unwrap_tuple: bool = False
):
    def hook(_module, _inputs, output):
        if unwrap_tuple and isinstance(output, tuple):
            output = output[0]
        if isinstance(output, torch.Tensor):
            update_max(target_dict, layer_idx, output)

    return hook


def _make_pre_hook(target_dict: dict[int, torch.Tensor], layer_idx: int):
    def hook(_module, inputs):
        if inputs and isinstance(inputs[0], torch.Tensor):
            update_max(target_dict, layer_idx, inputs[0])

    return hook


def _register_activation_hooks(
    decoder_layers,
    qkv_inputs: dict[int, torch.Tensor],
    o_proj_inputs: dict[int, torch.Tensor],
) -> list:
    print("[vLLM Stage 1] Registering activation hooks...")
    handles = []
    for layer_idx, layer in enumerate(decoder_layers):
        handles.append(
            layer.input_layernorm.register_forward_hook(
                _make_output_hook(qkv_inputs, layer_idx, unwrap_tuple=True)
            )
        )
        handles.append(
            layer.self_attn.qkv_proj.register_forward_pre_hook(
                _make_pre_hook(qkv_inputs, layer_idx)
            )
        )
        handles.append(
            layer.self_attn.attn.register_forward_hook(
                _make_output_hook(o_proj_inputs, layer_idx)
            )
        )
        handles.append(
            layer.self_attn.o_proj.register_forward_pre_hook(
                _make_pre_hook(o_proj_inputs, layer_idx)
            )
        )
    return handles


def _run_calibration(llm, prompts: list[dict[str, Any]], handles: list) -> None:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        detokenize=False,
        skip_special_tokens=False,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    print("[vLLM Stage 1] Running calibration prompts through vLLM...")
    try:
        llm.generate(prompts, sampling_params=sampling_params, use_tqdm=tqdm)
    finally:
        for handle in handles:
            handle.remove()
        release_accelerator_cache()


def _reduce_recorded_activations(
    num_layers: int,
    qkv_inputs: dict[int, torch.Tensor],
    o_proj_inputs: dict[int, torch.Tensor],
) -> None:
    print("[vLLM Stage 1] Reducing activation statistics across ranks...")
    for layer_idx in range(num_layers):
        if layer_idx not in qkv_inputs:
            raise RuntimeError(f"Layer {layer_idx} qkv hook was never triggered.")
        if layer_idx not in o_proj_inputs:
            raise RuntimeError(f"Layer {layer_idx} o_proj hook was never triggered.")
        qkv_inputs[layer_idx] = reduce_max_in_place(qkv_inputs[layer_idx])
        o_proj_inputs[layer_idx] = gather_and_concat(o_proj_inputs[layer_idx])


def _record_and_reduce(
    llm, prompts: list[dict[str, Any]]
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], int]:
    model = get_worker_model(llm)
    decoder_layers = model.model.layers
    num_layers = len(decoder_layers)
    qkv_inputs: dict[int, torch.Tensor] = {}
    o_proj_inputs: dict[int, torch.Tensor] = {}
    handles = _register_activation_hooks(decoder_layers, qkv_inputs, o_proj_inputs)
    _run_calibration(llm, prompts, handles)
    _reduce_recorded_activations(num_layers, qkv_inputs, o_proj_inputs)
    return qkv_inputs, o_proj_inputs, num_layers


def _build_metadata(
    *,
    model_dir: str,
    calib_data_path: str,
    num_calib_data: int,
    seq_len: int,
    num_layers: int,
    vllm_repo_dir: str,
    vllm_patch_path: str | None,
    tensor_parallel_size: int,
    world_size: int,
    rank: int,
    executor_backend: str | None,
    max_num_seqs: int,
    effective_max_num_batched_tokens: int,
    enable_expert_parallel: bool,
    compilation_config_dict: dict[str, Any] | None,
    enforce_eager: bool,
) -> dict[str, Any]:
    return {
        "model_dir": model_dir,
        "calib_data_path": calib_data_path,
        "num_calib_data": num_calib_data,
        "seq_len": seq_len,
        "num_layers": num_layers,
        "scaling_layers": SCALING_LAYERS,
        "backend": "vllm",
        "vllm_repo_dir": str(Path(vllm_repo_dir).expanduser().resolve()),
        "vllm_patch_path": vllm_patch_path,
        "tensor_parallel_size": tensor_parallel_size,
        "world_size": world_size,
        "rank": rank,
        "executor_backend": executor_backend,
        "engine_max_model_len": seq_len + 1,
        "engine_max_num_seqs": max_num_seqs,
        "engine_max_num_batched_tokens": effective_max_num_batched_tokens,
        "enable_expert_parallel": enable_expert_parallel,
        "compilation_config": compilation_config_dict,
        "enforce_eager": enforce_eager,
        "vllm_ascend_enable_flashcomm1": os.environ.get(
            "VLLM_ASCEND_ENABLE_FLASHCOMM1"
        ),
        "notes": {
            "qkv_record_source": "input_layernorm output / qkv_proj input fallback",
            "o_proj_record_source": (
                "self_attn.attn output / o_proj input fallback "
                "concatenated across TP ranks"
            ),
            "decode_note": (
                "generation uses max_tokens=1, so prompt prefill dominates "
                "but one decode step is also executed"
            ),
        },
    }


def _save_results(
    output_dir: str,
    num_layers: int,
    qkv_inputs: dict[int, torch.Tensor],
    o_proj_inputs: dict[int, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    print(f"[vLLM Stage 1] Saving activation scales to {output_dir}...")
    save_recorded_scales(output_dir, num_layers, qkv_inputs, o_proj_inputs, metadata)
    print(f"[vLLM Stage 1] Done. Recorded {num_layers * 4} activation scale tensors.")


def record_activations_vllm(
    model_dir: str,
    calib_data_path: str,
    output_dir: str,
    num_calib_data: int = 512,
    seq_len: int = 2048,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    vllm_repo_dir: str = DEFAULT_VLLM_REPO_DIR,
    vllm_patch_path: str | None = DEFAULT_VLLM_PATCH_PATH,
    enable_expert_parallel: bool = True,
    max_num_seqs: int = 32,
    max_num_batched_tokens: int = 32768,
    compilation_config: str | None = DEFAULT_COMPILATION_CONFIG,
    enforce_eager: bool = False,
    flashcomm1: str = "1",
) -> None:
    prepare_vllm_runtime(vllm_repo_dir, vllm_patch_path, flashcomm1)

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _validate_world_size(env_world_size, tensor_parallel_size)
    compilation_config_dict = parse_compilation_config(compilation_config)
    effective_max_num_batched_tokens = _resolve_max_num_batched_tokens(
        max_num_batched_tokens, seq_len
    )
    prompts = _load_tokenizer_and_prompts(
        model_dir, calib_data_path, num_calib_data, seq_len
    )
    executor_backend = "external_launcher" if env_world_size > 1 else None

    llm = _build_vllm_engine(
        model_dir=model_dir,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        seq_len=seq_len,
        enforce_eager=enforce_eager,
        max_num_seqs=max_num_seqs,
        effective_max_num_batched_tokens=effective_max_num_batched_tokens,
        enable_expert_parallel=enable_expert_parallel,
        compilation_config_dict=compilation_config_dict,
        executor_backend=executor_backend,
    )
    qkv_inputs, o_proj_inputs, num_layers = _record_and_reduce(llm, prompts)
    rank = get_rank()
    world_size = get_world_size()
    barrier()

    if rank == 0:
        metadata = _build_metadata(
            model_dir=model_dir,
            calib_data_path=calib_data_path,
            num_calib_data=num_calib_data,
            seq_len=seq_len,
            num_layers=num_layers,
            vllm_repo_dir=vllm_repo_dir,
            vllm_patch_path=vllm_patch_path,
            tensor_parallel_size=tensor_parallel_size,
            world_size=world_size,
            rank=rank,
            executor_backend=executor_backend,
            max_num_seqs=max_num_seqs,
            effective_max_num_batched_tokens=effective_max_num_batched_tokens,
            enable_expert_parallel=enable_expert_parallel,
            compilation_config_dict=compilation_config_dict,
            enforce_eager=enforce_eager,
        )
        _save_results(output_dir, num_layers, qkv_inputs, o_proj_inputs, metadata)

    barrier()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1: record activation statistics via vLLM"
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--calib_data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_calib_data", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument(
        "--vllm_repo_dir",
        default=os.environ.get("VLLM_REPO_DIR", DEFAULT_VLLM_REPO_DIR),
    )
    parser.add_argument(
        "--vllm_patch_path",
        default=os.environ.get("VLLM_PATCH_PATH", DEFAULT_VLLM_PATCH_PATH),
    )
    parser.add_argument(
        "--enable_expert_parallel",
        type=int,
        choices=(0, 1),
        default=int(os.environ.get("ENABLE_EXPERT_PARALLEL", "1")),
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=int(os.environ.get("VLLM_MAX_NUM_SEQS", "32")),
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=int(os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "32768")),
    )
    parser.add_argument(
        "--compilation_config",
        default=os.environ.get("VLLM_COMPILATION_CONFIG", DEFAULT_COMPILATION_CONFIG),
    )
    parser.add_argument(
        "--enforce_eager",
        type=int,
        choices=(0, 1),
        default=int(os.environ.get("VLLM_ENFORCE_EAGER", "0")),
    )
    parser.add_argument(
        "--flashcomm1",
        default=os.environ.get("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1"),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    record_activations_vllm(
        model_dir=args.model_dir,
        calib_data_path=args.calib_data_path,
        output_dir=args.output_dir,
        num_calib_data=args.num_calib_data,
        seq_len=args.seq_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_repo_dir=args.vllm_repo_dir,
        vllm_patch_path=args.vllm_patch_path,
        enable_expert_parallel=bool(args.enable_expert_parallel),
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        compilation_config=args.compilation_config,
        enforce_eager=bool(args.enforce_eager),
        flashcomm1=args.flashcomm1,
    )


if __name__ == "__main__":
    main()
