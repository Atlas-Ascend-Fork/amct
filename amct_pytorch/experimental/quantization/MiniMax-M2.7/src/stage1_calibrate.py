"""
OSPlus SmoothQuant Stage 1：基于 HuggingFace Transformers 在 NPU 上完成校准与
scale 搜索（MiniMax-M2.7，W4A4 MXFP4 目标）。

本阶段以 BF16 模型为后端做单批 eager 前向，通过 forward hook 在两类位置收集校准
激活，再逐层运行 OSPlus 阈值搜索，为每层产出两组 SmoothQuant 等价缩放 scale：
  - Group 1：介于 ``input_layernorm`` 与 q/k/v 投影之间，沿隐藏维（hidden_size）缩放；
  - Group 2：介于 v 投影与 o 投影之间，沿注意力查询维（num_attn_heads × head_dim）
    缩放，并在搜索时显式建模 GQA 折叠约束（同一 KV 头内的重复查询头共享 scale）。

搜索目标为 W4A4 MXFP4 伪量化后的逐层输出重建误差（详见 ``mxfp4_fake_quant.py``），
因此得到的最优 scale 直接面向部署侧的实际数值格式。

Hook 点：
  - ``layer.input_layernorm`` forward hook       -> Group 1 激活 [N, hidden]
  - ``layer.self_attn.o_proj`` forward_pre_hook   -> Group 2 激活 [N, q_size]

输出（与 Stage 2 融合导出严格对齐）：
  record_dir/
    layer_{i}_attn_scale.pt    : torch.Tensor[hidden_size]
    layer_{i}_oproj_scale.pt   : torch.Tensor[num_attn_heads * head_dim]  (GQA folded)
    activations/layer_{i}_attn_act.pt    : torch.Tensor[N, hidden] (bf16)
    activations/layer_{i}_oproj_act.pt   : torch.Tensor[N, q_size] (bf16)
    metadata.json

NPU 相关适配：
  - 在构建模型（``device_map="auto"``）之前 ``import torch_npu``，使 accelerate
    能识别 NPU 设备；
  - 用 accelerator-agnostic 的 ``_release_accelerator_cache()`` 释放显存，
    NPU 上调用 ``torch.npu.empty_cache()``。
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import random
from collections import defaultdict  # noqa: F401
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, Semaphore

import torch

# IMPORTANT: torch_npu must be imported *before* HF builds the model with
# device_map="auto", otherwise accelerate cannot see the NPU devices.
try:
    import torch_npu  # noqa: F401  (registers the npu backend on import)
except ImportError:
    torch_npu = None  # type: ignore[assignment]

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mxfp4_fake_quant import fake_quantize_mxfp4_activation, fake_quantize_mxfp4_weight  # noqa: E402

logger = logging.getLogger("OS+SQ-HF-NPU")
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")


# ── MiniMax-M2 architecture constants ──
NUM_ATTENTION_HEADS = 48
NUM_KEY_VALUE_HEADS = 8
HIDDEN_SIZE = 3072
HEAD_DIM = 128
NUM_HEAD_REPEATS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 6
NUM_SEARCH_STEPS = 200


# ── NPU-aware empty_cache (the only piece of NPU-specific glue) ──────────────
def _release_accelerator_cache() -> None:
    """``torch.cuda.empty_cache()`` equivalent that works on NPU.

    在删除每层激活 / 权重后调用，用于把 accelerator 上的工作集控制在有限范围内。
    """
    if torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Calibration data loading ──────────────────
def load_calib_data(
    calib_data_path: str,
    model_dir: str,
    num_samples: int,
    seq_len: int,
) -> list[torch.Tensor]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, padding_side="left"
    )

    prompts: list[str] = []
    with open(calib_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= num_samples:
                break
            try:
                obj = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            messages = obj.get("messages")
            if messages:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                prompts.append(text)
            elif "text" in obj:
                prompts.append(obj["text"])

    random.shuffle(prompts)

    input_ids_list = []
    for prompt in prompts:
        ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=seq_len
        ).input_ids[0]
        input_ids_list.append(ids)

    logger.info(f"Loaded {len(input_ids_list)} calibration samples (seq_len={seq_len})")
    return input_ids_list


# ── OSPlus Migration Search ──────────────────────
class OSPlusMigrator:
    """面向 MXFP4 量化的 OSPlus 单参数阈值搜索。"""

    def __init__(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        num_steps: int = NUM_SEARCH_STEPS,
    ):
        self.activation = activation.float()
        self.weight = weight.float()
        self.num_steps = num_steps
        self.device = activation.device

        self.cmx = self.activation.max(dim=0)[0]
        self.cmn = self.activation.min(dim=0)[0]
        self.amx = max(self.activation.max().item(), 0.0)
        self.amn = min(self.activation.min().item(), 0.0)
        self.fp_output = torch.matmul(self.activation, self.weight.T)

    def search(self) -> torch.Tensor:
        bounds_max = max(-self.amn, self.amx)
        bounds_min = 0.1
        step = (bounds_max - bounds_min) / self.num_steps

        best_loss = None
        best_scale = None
        st = bounds_max
        for _ in range(self.num_steps):
            scale = self._compute_scale(st)
            loss = self._compute_loss(scale)
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_scale = scale.clone()
            st -= step
            if st < bounds_min:
                break

        logger.info(
            f"  best_loss={best_loss:.4f}, scale range=[{best_scale.min():.3f}, {best_scale.max():.3f}]"
        )
        return best_scale

    def _compute_scale(self, st: float) -> torch.Tensor:
        st_tensor = torch.tensor(st, dtype=torch.float32, device=self.device)
        mx_scale = torch.where(
            self.cmx > st_tensor, self.cmx / st_tensor, torch.ones_like(self.cmx)
        )
        mn_scale = torch.where(
            self.cmn < -st_tensor, self.cmn / (-st_tensor), torch.ones_like(self.cmn)
        )
        return torch.max(mx_scale, mn_scale)

    def _compute_loss(self, scale: torch.Tensor) -> float:
        scaled_act = self.activation / scale.unsqueeze(0)
        scaled_weight = self.weight * scale.unsqueeze(0)
        q_act = fake_quantize_mxfp4_activation(scaled_act)
        q_weight = fake_quantize_mxfp4_weight(scaled_weight)
        q_output = torch.matmul(q_act, q_weight.T)
        loss = (q_output - self.fp_output).pow(2).sum(dim=-1).mean()
        return loss.item()


class OSPlusMigratorGQA(OSPlusMigrator):
    """带 GQA 折叠约束的 OSPlus 搜索。"""

    def __init__(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        num_kv_heads: int = NUM_KEY_VALUE_HEADS,
        num_attention_heads: int = NUM_ATTENTION_HEADS,
        num_steps: int = NUM_SEARCH_STEPS,
    ):
        super().__init__(activation, weight, num_steps)
        self.num_kv_heads = num_kv_heads
        self.num_attention_heads = num_attention_heads
        self.num_repeats = num_attention_heads // num_kv_heads
        self.head_dim = activation.shape[-1] // num_attention_heads

    def search(self) -> torch.Tensor:
        bounds_max = max(-self.amn, self.amx)
        bounds_min = 0.1
        step = (bounds_max - bounds_min) / self.num_steps

        best_loss = None
        best_scale = None
        st = bounds_max
        for _ in range(self.num_steps):
            raw_scale = self._compute_scale(st)
            folded = self._fold_scale_gqa(raw_scale)
            loss = self._compute_loss(raw_scale)
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_scale = folded.clone()
            st -= step
            if st < bounds_min:
                break

        logger.info(
            f"  best_loss={best_loss:.4f}, scale range=[{best_scale.min():.3f}, {best_scale.max():.3f}]"
        )
        return best_scale

    def _fold_scale_gqa(self, scale: torch.Tensor) -> torch.Tensor:
        grouped = scale.view(self.num_kv_heads, self.num_repeats, self.head_dim)
        kv_scale = grouped.max(dim=1, keepdim=True)[0]
        return kv_scale.expand(
            self.num_kv_heads, self.num_repeats, self.head_dim
        ).reshape(-1)

    def _compute_loss(self, scale: torch.Tensor) -> float:
        folded = self._fold_scale_gqa(scale)
        scaled_act = self.activation / folded.unsqueeze(0)
        scaled_weight = self.weight * folded.unsqueeze(0)
        q_act = fake_quantize_mxfp4_activation(scaled_act)
        q_weight = fake_quantize_mxfp4_weight(scaled_weight)
        q_output = torch.matmul(q_act, q_weight.T)
        loss = (q_output - self.fp_output).pow(2).sum(dim=-1).mean()
        return loss.item()


# ── Collect ALL layers' activations in ONE forward pass ───────────
def get_module(model, path: str):
    module = model
    for part in path.split("."):
        module = getattr(module, part)
    return module


def _append_capped_chunk(
    buffers: dict[int, list[torch.Tensor]],
    tokens: dict[int, int],
    layer_idx: int,
    chunk: torch.Tensor,
    max_tokens: int,
) -> None:
    remaining = max_tokens - tokens[layer_idx]
    if remaining <= 0:
        return
    if chunk.shape[0] > remaining:
        chunk = chunk[:remaining]
    buffers[layer_idx].append(chunk)
    tokens[layer_idx] += chunk.shape[0]


def _make_attn_hook(
    layer_idx: int,
    attn_buffers: dict[int, list[torch.Tensor]],
    attn_tokens: dict[int, int],
    max_tokens: int,
    lock: Lock,
):
    def hook_fn(module, _input, output):
        with lock:
            if attn_tokens[layer_idx] >= max_tokens:
                return
            chunk = output.detach().reshape(-1, output.shape[-1]).cpu()
            _append_capped_chunk(
                attn_buffers, attn_tokens, layer_idx, chunk, max_tokens
            )

    return hook_fn


def _make_oproj_hook(
    layer_idx: int,
    oproj_buffers: dict[int, list[torch.Tensor]],
    oproj_tokens: dict[int, int],
    max_tokens: int,
    lock: Lock,
):
    def hook_fn(module, inputs):
        with lock:
            if oproj_tokens[layer_idx] >= max_tokens:
                return
            inp = inputs[0] if isinstance(inputs, tuple) else inputs
            chunk = inp.detach().reshape(-1, inp.shape[-1]).cpu()
            _append_capped_chunk(
                oproj_buffers, oproj_tokens, layer_idx, chunk, max_tokens
            )

    return hook_fn


def _register_activation_hooks(
    decoder_layers,
    num_layers: int,
    attn_buffers: dict[int, list[torch.Tensor]],
    oproj_buffers: dict[int, list[torch.Tensor]],
    attn_tokens: dict[int, int],
    oproj_tokens: dict[int, int],
    max_tokens: int,
    lock: Lock,
):
    handles = []
    for i in range(num_layers):
        layer = decoder_layers[i]
        handles.append(
            layer.input_layernorm.register_forward_hook(
                _make_attn_hook(i, attn_buffers, attn_tokens, max_tokens, lock)
            )
        )
        handles.append(
            layer.self_attn.o_proj.register_forward_pre_hook(
                _make_oproj_hook(i, oproj_buffers, oproj_tokens, max_tokens, lock)
            )
        )
    return handles


def _resolve_model_device(model, decoder_layers) -> torch.device:
    try:
        return decoder_layers[0].input_layernorm.weight.device
    except Exception:
        return next(model.parameters()).device


def _run_activation_forward(
    model,
    decoder_layers,
    input_ids_list: list[torch.Tensor],
    batch_size: int,
    num_layers: int,
    attn_tokens: dict[int, int],
    oproj_tokens: dict[int, int],
    max_tokens: int,
) -> None:
    model.eval()
    with torch.no_grad():
        for i in tqdm(
            range(0, len(input_ids_list), batch_size), desc="Collecting activations"
        ):
            all_full = all(
                attn_tokens[j] >= max_tokens and oproj_tokens[j] >= max_tokens
                for j in range(num_layers)
            )
            if all_full:
                logger.info("All layers reached max_tokens budget, stopping early.")
                break

            end = i + batch_size
            batch_ids = input_ids_list[i:end]
            max_len = max(ids.shape[0] for ids in batch_ids)
            padded = torch.zeros(len(batch_ids), max_len, dtype=torch.long)
            for j, ids in enumerate(batch_ids):
                padded[j, : ids.shape[0]] = ids

            device = _resolve_model_device(model, decoder_layers)
            padded = padded.to(device)
            model(padded, use_cache=False)


def _finalize_activation_buffers(
    num_layers: int,
    attn_buffers: dict[int, list[torch.Tensor]],
    oproj_buffers: dict[int, list[torch.Tensor]],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    attn_acts = {}
    oproj_acts = {}
    for i in range(num_layers):
        attn_acts[i] = (
            torch.cat(attn_buffers[i], dim=0)
            if attn_buffers[i]
            else torch.zeros(0, HIDDEN_SIZE)
        )
        oproj_acts[i] = (
            torch.cat(oproj_buffers[i], dim=0)
            if oproj_buffers[i]
            else torch.zeros(0, HIDDEN_SIZE)
        )
    return attn_acts, oproj_acts


def _init_activation_buffers(
    num_layers: int,
) -> tuple[
    dict[int, list[torch.Tensor]],
    dict[int, list[torch.Tensor]],
    dict[int, int],
    dict[int, int],
]:
    attn_buffers = {i: [] for i in range(num_layers)}
    oproj_buffers = {i: [] for i in range(num_layers)}
    attn_tokens = {i: 0 for i in range(num_layers)}
    oproj_tokens = {i: 0 for i in range(num_layers)}
    return attn_buffers, oproj_buffers, attn_tokens, oproj_tokens


def _log_collected_activations(
    attn_acts: dict[int, torch.Tensor],
    oproj_acts: dict[int, torch.Tensor],
    max_tokens: int,
) -> None:
    attn0 = attn_acts.get(0)
    oproj0 = oproj_acts.get(0)
    if attn0 is None or oproj0 is None:
        raise KeyError("Expected layer-0 activations after collection")
    logger.info(
        f"Activations collected: attn={attn0.shape[0]} tokens, "
        f"oproj={oproj0.shape[0]} tokens per layer (capped at {max_tokens})"
    )


def collect_all_activations(
    model,
    num_layers: int,
    input_ids_list: list[torch.Tensor],
    batch_size: int = 2,
    max_tokens: int = 4096,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """在所有层挂 hook，一次前向同时收集 input_layernorm 输出（Group 1）与
    o_proj 输入（Group 2）激活。

    在 NPU 上，``device`` 会解析为 ``npu:*``（因模型以 NPU-aware ``device_map`` 加载）。
    """
    decoder_layers = get_module(model, "model.layers")
    attn_buffers, oproj_buffers, attn_tokens, oproj_tokens = _init_activation_buffers(
        num_layers
    )
    lock = Lock()
    handles = _register_activation_hooks(
        decoder_layers,
        num_layers,
        attn_buffers,
        oproj_buffers,
        attn_tokens,
        oproj_tokens,
        max_tokens,
        lock,
    )
    logger.info(
        f"Registered hooks on all {num_layers} layers "
        f"(input_layernorm + o_proj), running forward pass..."
    )
    _run_activation_forward(
        model,
        decoder_layers,
        input_ids_list,
        batch_size,
        num_layers,
        attn_tokens,
        oproj_tokens,
        max_tokens,
    )
    for h in handles:
        h.remove()
    attn_acts, oproj_acts = _finalize_activation_buffers(
        num_layers, attn_buffers, oproj_buffers
    )
    del attn_buffers, oproj_buffers
    gc.collect()
    _log_collected_activations(attn_acts, oproj_acts, max_tokens)
    return attn_acts, oproj_acts


# ── Activation cache (save/load) ──────────────────────────────────
def _act_dir(record_dir: str) -> str:
    return os.path.join(record_dir, "activations")


def save_activations(
    record_dir: str,
    attn_acts: dict[int, torch.Tensor],
    oproj_acts: dict[int, torch.Tensor],
) -> None:
    act_dir = _act_dir(record_dir)
    os.makedirs(act_dir, exist_ok=True)
    for layer_idx in sorted(attn_acts.keys()):
        torch.save(
            attn_acts[layer_idx].bfloat16(),
            os.path.join(act_dir, f"layer_{layer_idx}_attn_act.pt"),
        )
        torch.save(
            oproj_acts[layer_idx].bfloat16(),
            os.path.join(act_dir, f"layer_{layer_idx}_oproj_act.pt"),
        )
    total_bytes = sum(
        t.numel() * t.element_size()
        for t in list(attn_acts.values()) + list(oproj_acts.values())
    )
    logger.info(
        f"Saved activations to {act_dir} ({total_bytes / 1e9:.2f} GB in memory, bf16 on disk)"
    )


def load_activations(
    record_dir: str,
    num_layers: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]] | None:
    act_dir = _act_dir(record_dir)
    if not os.path.isdir(act_dir):
        return None
    attn_acts = {}
    oproj_acts = {}
    for layer_idx in range(num_layers):
        attn_path = os.path.join(act_dir, f"layer_{layer_idx}_attn_act.pt")
        oproj_path = os.path.join(act_dir, f"layer_{layer_idx}_oproj_act.pt")
        if not os.path.isfile(attn_path) or not os.path.isfile(oproj_path):
            logger.warning(
                f"Activation cache incomplete (missing layer {layer_idx}), will re-collect."
            )
            return None
        attn_acts[layer_idx] = torch.load(
            attn_path, map_location="cpu", weights_only=True
        )
        oproj_acts[layer_idx] = torch.load(
            oproj_path, map_location="cpu", weights_only=True
        )
    attn0 = attn_acts.get(0)
    oproj0 = oproj_acts.get(0)
    if attn0 is None or oproj0 is None:
        raise KeyError("Expected layer-0 activations in activation cache")
    logger.info(
        f"Loaded cached activations from {act_dir}: "
        f"{num_layers} layers, attn={attn0.shape}, oproj={oproj0.shape}"
    )
    return attn_acts, oproj_acts


# ── Per-layer scale computation ────────────
_device_semaphores: dict[str, Semaphore] = {}
_device_sem_lock = Lock()


def _get_device_semaphore(device: torch.device) -> Semaphore:
    key = str(device)
    if key not in _device_semaphores:
        with _device_sem_lock:
            if key not in _device_semaphores:
                _device_semaphores[key] = Semaphore(1)
    return _device_semaphores[key]


def compute_and_save_layer_scales(
    model,
    layer_idx: int,
    attn_act: torch.Tensor,
    oproj_act: torch.Tensor,
    record_dir: str,
) -> int:
    """Compute OS+SQ scales for one layer and save to disk."""
    decoder_layers = get_module(model, "model.layers")
    layer = decoder_layers[layer_idx]

    device = get_module(layer, "self_attn.q_proj").weight.device
    sem = _get_device_semaphore(device)

    sem.acquire()
    try:
        q_w = get_module(layer, "self_attn.q_proj").weight.data.float()
        k_w = get_module(layer, "self_attn.k_proj").weight.data.float()
        v_w = get_module(layer, "self_attn.v_proj").weight.data.float()
        combined_qkv = torch.cat([q_w, k_w, v_w], dim=0)

        attn_act_dev = attn_act.to(device)

        logger.info(
            f"Layer {layer_idx} - Group 1 (Attention qkv): act={attn_act_dev.shape}, weight={combined_qkv.shape}"
        )
        attn_scale = OSPlusMigrator(attn_act_dev, combined_qkv).search()

        del attn_act_dev, combined_qkv, q_w, k_w, v_w
        gc.collect()
        _release_accelerator_cache()

        o_w = get_module(layer, "self_attn.o_proj").weight.data.float()
        oproj_act_dev = oproj_act.to(device)

        logger.info(
            f"Layer {layer_idx} - Group 2 (o_proj GQA): act={oproj_act_dev.shape}, weight={o_w.shape}"
        )
        oproj_scale = OSPlusMigratorGQA(
            oproj_act_dev,
            o_w,
            num_kv_heads=NUM_KEY_VALUE_HEADS,
            num_attention_heads=NUM_ATTENTION_HEADS,
        ).search()

        del oproj_act_dev, o_w

        torch.save(
            attn_scale.cpu(),
            os.path.join(record_dir, f"layer_{layer_idx}_attn_scale.pt"),
        )
        torch.save(
            oproj_scale.cpu(),
            os.path.join(record_dir, f"layer_{layer_idx}_oproj_scale.pt"),
        )

        del attn_scale, oproj_scale
        gc.collect()
        _release_accelerator_cache()
    finally:
        sem.release()

    return layer_idx


# ── Main ───────────────
def _parse_args():
    parser = argparse.ArgumentParser(
        description="OS+SQ Stage 1 (HF on NPU): Calibration and scale computation"
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--calib_data", required=True)
    parser.add_argument("--record_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=32768)
    parser.add_argument("--num_calib_data", type=int, default=512)
    parser.add_argument("--load_device_map", default="auto", choices=["cpu", "auto"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_tokens_per_layer", type=int, default=4096)
    parser.add_argument("--model_files_dir", default=None)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel threads for OS+ search",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip layers whose scale files already exist in record_dir",
    )
    parser.add_argument(
        "--record_only",
        action="store_true",
        help="Only record + cache activations; do not run OS+ search.",
    )
    parser.add_argument(
        "--max_layers_to_search",
        type=int,
        default=0,
        help="If >0, only search the first N layers (smoke test).",
    )
    return parser.parse_args()


def _load_stage1_model(model_dir: str, load_device_map: str):
    from transformers import AutoModelForCausalLM

    device_map = "auto" if load_device_map == "auto" else {"": "cpu"}
    logger.info(f"Loading model from {model_dir} with device_map={device_map}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def _layers_to_process(record_dir: str, num_layers: int, resume: bool) -> list[int]:
    layers = list(range(num_layers))
    if not resume:
        return layers
    skipped = []
    for i in list(layers):
        attn_path = os.path.join(record_dir, f"layer_{i}_attn_scale.pt")
        oproj_path = os.path.join(record_dir, f"layer_{i}_oproj_scale.pt")
        if os.path.isfile(attn_path) and os.path.isfile(oproj_path):
            skipped.append(i)
            layers.remove(i)
    if skipped:
        logger.info(
            f"Resuming: skipping {len(skipped)} already-completed layers: {skipped}"
        )
    return layers


def _prepare_activations(
    model,
    num_layers: int,
    input_ids_list: list[torch.Tensor],
    record_dir: str,
    batch_size: int,
    max_tokens_per_layer: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    cached = load_activations(record_dir, num_layers)
    if cached is not None:
        return cached

    logger.info(
        f"Phase 1: Collecting activations for all {num_layers} layers "
        f"in one forward pass..."
    )
    attn_acts, oproj_acts = collect_all_activations(
        model,
        num_layers,
        input_ids_list,
        batch_size=batch_size,
        max_tokens=max_tokens_per_layer,
    )
    gc.collect()
    save_activations(record_dir, attn_acts, oproj_acts)
    return attn_acts, oproj_acts


def _select_layers_for_search(
    layers_to_process: list[int], max_layers_to_search: int
) -> list[int]:
    layers_for_search = list(layers_to_process)
    if max_layers_to_search <= 0:
        return layers_for_search
    kept = layers_for_search[:max_layers_to_search]
    if len(kept) < len(layers_for_search):
        logger.info(
            f"Smoke mode: --max_layers_to_search={max_layers_to_search} "
            f"limits search to layers {kept}"
        )
    return kept


def _run_osplus_search(
    model,
    decoder_layers,
    layers_for_search: list[int],
    attn_acts: dict[int, torch.Tensor],
    oproj_acts: dict[int, torch.Tensor],
    record_dir: str,
    num_workers: int,
) -> None:
    devices_used = {
        str(get_module(decoder_layers[i], "self_attn.q_proj").weight.device)
        for i in layers_for_search
    }
    num_devs = len(devices_used)
    workers = min(num_workers, len(layers_for_search), max(num_devs, 1))
    logger.info(
        f"Phase 2: Running OS+SQ search for {len(layers_for_search)} layers "
        f"with {workers} workers across {num_devs} devices {devices_used}"
    )

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                compute_and_save_layer_scales,
                model,
                layer_idx,
                attn_acts[layer_idx],
                oproj_acts[layer_idx],
                record_dir,
            ): layer_idx
            for layer_idx in layers_for_search
        }
        for fut in as_completed(futures):
            layer_idx = futures[fut]
            try:
                fut.result()
                completed += 1
                logger.info(
                    f"  [{completed}/{len(layers_for_search)}] Layer {layer_idx} done."
                )
            except Exception as e:
                logger.error(f"  Layer {layer_idx} FAILED: {e}", exc_info=True)


def _write_stage1_metadata(args, num_layers: int) -> None:
    metadata = {
        "model_dir": args.model_dir,
        "calib_data": args.calib_data,
        "num_calib_data": args.num_calib_data,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "num_layers": num_layers,
        "num_search_steps": NUM_SEARCH_STEPS,
        "max_tokens_per_layer": args.max_tokens_per_layer,
        "algorithm": "osplus_sq",
        "quantization_target": "mxfp4_w4a4",
        "implementation": "huggingface_transformers_on_npu",
        "scale_groups": {
            "group1": "input_layernorm \u2192 q/k/v_proj (OS+ search)",
            "group2": "v_proj \u2192 o_proj (OS+ search with GQA constraint)",
        },
        "gqa_config": {
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "num_key_value_heads": NUM_KEY_VALUE_HEADS,
            "head_dim": HEAD_DIM,
            "num_repeats": NUM_HEAD_REPEATS,
        },
        "parallel": True,
    }
    with open(os.path.join(args.record_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    args = _parse_args()
    os.makedirs(args.record_dir, exist_ok=True)

    tokenizer_dir = args.model_files_dir if args.model_files_dir else args.model_dir
    logger.info(f"Loading calibration data from {args.calib_data}...")
    input_ids_list = load_calib_data(
        args.calib_data, tokenizer_dir, args.num_calib_data, args.seq_len
    )

    model = _load_stage1_model(args.model_dir, args.load_device_map)
    decoder_layers = get_module(model, "model.layers")
    num_layers = len(decoder_layers)
    logger.info(f"Model loaded: {num_layers} decoder layers")

    layers = _layers_to_process(args.record_dir, num_layers, args.resume)
    if not layers and not args.record_only:
        logger.info("All layers already completed. Nothing to do.")
    else:
        attn_acts, oproj_acts = _prepare_activations(
            model,
            num_layers,
            input_ids_list,
            args.record_dir,
            args.batch_size,
            args.max_tokens_per_layer,
        )
        del input_ids_list

        if args.record_only:
            logger.info(
                "record_only=1: activations cached; skipping OS+ search and metadata write."
            )
            return

        layers_for_search = _select_layers_for_search(layers, args.max_layers_to_search)
        _run_osplus_search(
            model,
            decoder_layers,
            layers_for_search,
            attn_acts,
            oproj_acts,
            args.record_dir,
            args.num_workers,
        )
        del attn_acts, oproj_acts
        gc.collect()

    _write_stage1_metadata(args, num_layers)
    logger.info(f"Done. Scales saved to {args.record_dir}")


if __name__ == "__main__":
    main()
