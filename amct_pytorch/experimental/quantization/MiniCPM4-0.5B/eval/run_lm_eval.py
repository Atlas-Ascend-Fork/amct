"""
lm_eval wrapper（NPU 兼容）：
  1. 导入 torch_npu，注册 NPU 设备
  2. 修复 transformers caching_allocator_warmup 调用原生 CUDA API
  3. 修复 GenerationConfig.from_dict 将 torch.dtype 写入 JSON 报错
  4. 修复 modeling_minicpm past_key_values 格式检查（C/D）：
     lm_eval 做 loglikelihood 时不使用 KV cache，关掉 use_cache 即可绕过
     模型内部的 "must use Cache class" 检查，无需修改模型文件
  5. 修复 lm_eval 以 dtype= 调用 from_pretrained 而 transformers ≥4.48 签名为
     torch_dtype=，导致 dtype 流入 model_kwargs 并传给 cls.__init__() 报 TypeError
  6. QuIP CUDA 模型用 load_quantized_model 加载（hf_cuda 有 quip_params，
     标准 AutoModelForCausalLM 无法正确还原量化层）
"""
import sys
from pathlib import Path
import torch
import torch_npu  # noqa: F401 — 必须先于 transformers 导入

# ── 修复 1：caching_allocator_warmup ────────────────────────────────────
import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **kw: None

# ── 修复 2：torch.cuda.mem_get_info（保底） ──────────────────────────────
def _npu_mem_get_info(device=0):
    try:
        props = torch_npu.npu.get_device_properties(device)
        total = props.total_memory
        used  = torch_npu.npu.memory_allocated(device)
        return total - used, total
    except Exception:
        return 1 << 30, 1 << 30
torch.cuda.mem_get_info = _npu_mem_get_info

# ── 修复 3：GenerationConfig.from_dict 收到 torch.dtype 后 JSON 序列化失败
from transformers.generation.configuration_utils import GenerationConfig as _GC
_orig_from_dict = _GC.from_dict.__func__

@classmethod  # type: ignore[misc]
def _patched_from_dict(cls, config_dict, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, torch.dtype)}
    return _orig_from_dict(cls, config_dict, **kwargs)

_GC.from_dict = _patched_from_dict

# ── 修复 5：lm_eval 以 dtype= 调用 from_pretrained，transformers 不识别它 ──
# transformers 的签名是 torch_dtype=，dtype= 会落入 **kwargs → model_kwargs，
# 最终传给 cls.__init__() 导致 TypeError。
# 在 PreTrainedModel.from_pretrained 入口将 dtype= 转换为 torch_dtype=。
import transformers as _tf
_orig_fp = _tf.PreTrainedModel.from_pretrained.__func__

@classmethod  # type: ignore[misc]
def _patched_fp(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    if 'dtype' in kwargs and 'torch_dtype' not in kwargs:
        kwargs['torch_dtype'] = kwargs.pop('dtype')
    elif 'dtype' in kwargs:
        kwargs.pop('dtype')
    return _orig_fp(cls, pretrained_model_name_or_path, *model_args, **kwargs)

_tf.PreTrainedModel.from_pretrained = _patched_fp

# ── 修复 4（C/D）：模型 forward 中 past_key_values 格式检查 ─────────────
# lm_eval 的 loglikelihood 全序列一次性推理，不需要 KV cache。
# 加载后禁用 use_cache，绕过 modeling_minicpm 里的 "must use Cache class" 断言。
# 修复 6：QuIP 模型（hf_cuda/hf_npu_had）保存的是 QuantizedLinear 而非标准
# nn.Linear，没有 weight tensor，必须走 format_quantize_model + 量化权重加载。
# CUDA 上用 load_checkpoint_and_dispatch（单卡直接 device_map={"":"cuda:0"}），
# NPU 上同理用 {"":"npu:0"}，均不依赖 infer_auto_device_map。
from lm_eval.models.huggingface import HFLM as _HFLM
from transformers import AutoConfig as _AutoConfig

_orig_create = _HFLM._create_model

def _load_quip_model(pretrained, device_str, torch_dtype, trust_remote_code):
    """正确加载 QuIP 量化模型（format_quantize_model + state_dict 直接加载）。"""
    from pathlib import Path as _Path
    import glob as _glob
    _repo_root = Path(__file__).resolve().parents[1]
    for _p in (str(_repo_root / "create_init_ckpt"), str(_repo_root)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from model.general_model import format_quantize_model

    config = _AutoConfig.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
    # 'auto' 是字符串，from_config 不接受，统一转为 float16
    _resolved_dtype = (
        torch.float16
        if not isinstance(torch_dtype, torch.dtype)
        else torch_dtype
    )
    model = _tf.AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=trust_remote_code,
        torch_dtype=_resolved_dtype,
    )
    format_quantize_model(model=model, config=config)
    if config.tie_word_embeddings:
        model.tie_weights()

    # 加载 checkpoint（支持 safetensors 和 pytorch bin）
    state_dict = {}
    sf_files = sorted(_glob.glob(f"{pretrained}/*.safetensors"))
    if sf_files:
        from safetensors.torch import load_file as _sf_load
        for sf in sf_files:
            state_dict.update(_sf_load(sf, device="cpu"))
    else:
        bin_files = sorted(_glob.glob(f"{pretrained}/pytorch_model*.bin"))
        for bf in bin_files:
            state_dict.update(torch.load(bf, map_location="cpu"))

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device_str)
    return model

def _patched_create(self, pretrained, *args, **kwargs):
    _is_quip = False
    _device_str = None
    try:
        _cfg = _AutoConfig.from_pretrained(
            pretrained,
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )
        if hasattr(_cfg, "quip_params"):
            if torch.cuda.is_available():
                _is_quip, _device_str = True, "cuda:0"
            elif hasattr(torch, "npu") and torch.npu.is_available():
                _is_quip, _device_str = True, "npu:0"
    except Exception:
        pass

    if _is_quip:
        _dtype_arg = kwargs.get("dtype", "auto")
        _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        _torch_dtype = (
            _dtype_map.get(_dtype_arg, torch.float16)
            if isinstance(_dtype_arg, str)
            else _dtype_arg
        )
        self._model = _load_quip_model(
            pretrained=pretrained,
            device_str=_device_str,
            torch_dtype=_torch_dtype,
            trust_remote_code=kwargs.get("trust_remote_code", True),
        )
    else:
        _orig_create(self, pretrained, *args, **kwargs)

    # NPU 上 modeling_minicpm 有 "must use Cache class" 断言，必须禁用 use_cache。
    # CUDA 上可以保留 use_cache（generate_until 任务需要）。
    if _device_str is not None and "npu" in _device_str:
        self._model.config.use_cache = False
    elif not _is_quip:
        self._model.config.use_cache = False

_HFLM._create_model = _patched_create

from lm_eval.__main__ import cli_evaluate  # noqa: E402

sys.exit(cli_evaluate())
