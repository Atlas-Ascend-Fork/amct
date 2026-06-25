# Adapted from QuIP# (Cornell-RelaxML/quip-sharp), licensed under GPLv3.
# Source: https://github.com/Cornell-RelaxML/quip-sharp/blob/main/quantize_llama/quantize_finetune_llama.py
# Modified for MiniCPM quantization on Ascend NPU.
import os
import glog
import argparse
glog.setLevel(glog.INFO)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import json
import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers.cache_utils as _cache_utils
if not hasattr(_cache_utils, "CacheLayerMixin"):
    class CacheLayerMixin: pass
    _cache_utils.CacheLayerMixin = CacheLayerMixin
if not hasattr(_cache_utils, "DynamicLayer"):
    class DynamicLayer: pass
    _cache_utils.DynamicLayer = DynamicLayer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from lib import codebook, utils
from lib.algo import finetune


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=384, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--save_path', type=str)
parser.add_argument('--hessian_path', type=str)
parser.add_argument('--base_model', type=str)
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--sigma_reg2', default=1e-2, type=float)
parser.add_argument('--incoh_mode', default='had', type=str, choices=['had', 'kron'])
parser.add_argument('--lora_rank',default=0, type=int, help='if <=0 then turned off')
parser.add_argument('--scale_override', default=-1, type=float)
parser.add_argument('--scale_search_iters', default=1, type=int)
parser.add_argument('--resid_scale_override', default=-1, type=float)
parser.add_argument('--codebook', type=str)
parser.add_argument('--quip_tune_iters', default=10, type=int)
parser.add_argument('--use_fp64', action='store_true')
parser.add_argument('--full_svd', action='store_true')
parser.add_argument('--no_use_buffered', action='store_true')
parser.add_argument('--rescale_WH', action='store_true')
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--lowmem_ldlq', action='store_true')
parser.add_argument('--ft_lr', default=5e-5, type=float)
parser.add_argument('--ft_susv_lr', default=5e-4, type=float)
parser.add_argument('--ft_bs', default=4, type=int)
parser.add_argument('--ft_update_freq', default=2, type=int)
parser.add_argument('--ft_epochs', default=5, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=3, type=int)
parser.add_argument('--ft_grad_ckpt', action='store_true')
parser.add_argument('--dataset_path', default="/data", type=str)
parser.add_argument('--model_dtype', default='auto', choices=['auto', 'bfloat16', 'float16', 'float32'], type=str)


def check_exist(idx, args):
    suffix = ['qkv', 'o', 'up', 'down', 'layernorm']
    for _ in suffix:
        test = f'{args.save_path}/{idx}_{_}.pt'
        if not os.path.exists(test):
            return False
    return True


def quantize_llama_layer(layer, idx, cb, args, device, pre_orig_emb, orig_emb, position_embeddings, position_ids):
    if check_exist(idx, args):
        print(f"{idx} exits")
        return
    if hasattr(layer.self_attn, "q_proj"):
        quant_order = [
            ['self_attn.v_proj', 'qkv'],
            ['self_attn.q_proj', 'qkv'],
            ['self_attn.k_proj', 'qkv'],
            ['self_attn.o_proj', 'o'],
            ['mlp.up_proj', 'upgate'],
            ['mlp.gate_proj', 'upgate'],
            ['mlp.down_proj', 'down']
        ]
    else:
        quant_order = [
            ['self_attn.qkv_proj', 'qkv'],
            ['self_attn.o_proj', 'o'],
            ['mlp.gate_up_proj', 'upgate'],
            ['mlp.down_proj', 'down']
        ]

    finetune.quantize_finetune_decoder_layer(
        mixed_layer=layer, quant_order=quant_order, idx=idx,
        cb=cb, args=args, device=device,
        pre_orig_emb=pre_orig_emb, orig_emb=orig_emb,
        position_embeddings=position_embeddings, position_ids=position_ids)

    torch.save({
            'input_layernorm': layer.input_layernorm.weight,
            'post_attention_layernorm': layer.post_attention_layernorm.weight,
        }, f'{args.save_path}/{idx}_layernorm.pt'
    )
    del layer


def main(args):
    # glog.info("into main")
    cb = codebook.get_codebook(args.codebook)
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    if "minicpm" in config.architectures[0].lower():
        config._attn_implementation = "sdpa"
    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        torch_dtype=dtype_map[args.model_dtype],
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    glog.info(f"stage2 model dtype mode: {args.model_dtype}")
    # glog.info("model loaded")
    # save configs
    all_config = {'quant_args': args, 'model_config': model.config}
    quip_params = {
        'lora_rank': args.lora_rank,
        'rescale_WH': args.rescale_WH,
        'codebook': args.codebook,
        'codebook_version': cb.version,
        'codesz': cb.codesz,
        'idx_dtype': str(cb.idx_dtype),
        'packsz': cb.packsz,
        'resid_scale_override': args.resid_scale_override,
        'codebook_scale': getattr(cb, "codebook_scale", 1.0)
    }
    all_config['model_config'].update({'quip_params': quip_params})
    torch.save(all_config, os.path.join(args.save_path, 'config.pt'))
    # glog.info("config saved")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # glog.info('loaded model')

    if "pajama" in args.dataset_path:
        devset = utils.sample_rp1t_concat(tokenizer, args.devset_size, args.ctx_size, nproc=args.sample_proc)
    elif "jsonl" in args.dataset_path:
        devset = utils.sample_jsonl_concat(args.dataset_path, tokenizer, args.devset_size, args.ctx_size, nproc=args.sample_proc)
    else:
        not NotImplementedError(args.dataset_path)
    # glog.info('loaded dataset and devset')

    first_input = model.model.embed_tokens(devset) # [dataset_shape, seq_len, hidden_size]
    if hasattr(model.config, "scale_emb"):
        first_input = first_input * model.config.scale_emb

    # Embedding cache holds the per-layer input/output activations for the whole
    # devset: two full [devset, ctx, hidden] tensors. At devset=4096/ctx=2048 fp32
    # (3B) that is ~85.9GB EACH (~172GB combined) resident. Two backings:
    #   - in-RAM (default, HARNESS_EMB_MEMMAP=0): matches the historically-working
    #     3B run (job 492109), which ran memmap OFF — its note: "memmap path hit a
    #     non-deterministic segfault; in-RAM is simpler". Needs container headroom.
    #   - memmap (HARNESS_EMB_MEMMAP=1): np.memmap on local disk so the OS pages in
    #     only active per-batch slices; resident RAM drops to a few GB. Numerically
    #     identical (same bytes). WARNING: if the dir is tmpfs (RAM-backed, common
    #     for /tmp in containers) this saves nothing — point HARNESS_EMB_MEMMAP_DIR
    #     at a real disk.
    _emb_memmap = os.environ.get("HARNESS_EMB_MEMMAP", "0") != "0"
    # Revert the copy_/del D2H write and the reference-swap to the pre-memfix
    # .cpu()+clone() behaviour. memfix (c9fbe28) was needed for 3B OOM but is a
    # suspect for the 1B fp16 PPL=140 regression (job 499083); 1B has ample RAM
    # so it does not need memfix. Default 0 = memfix (3B-safe). Incompatible with
    # memmap (legacy clone materialises a real tensor); only meaningful in-RAM.
    _emb_legacy = os.environ.get("HARNESS_EMB_LEGACY_COPY", "0") != "0"
    if _emb_memmap:
        _np_dtype = {torch.float16: np.float16, torch.float32: np.float32}.get(
            first_input.dtype, np.float32)
        _emb_dir = os.environ.get("HARNESS_EMB_MEMMAP_DIR", "/tmp/emb_cache")
        os.makedirs(_emb_dir, exist_ok=True)
        _shape = tuple(first_input.shape)
        _fi_np = first_input.to(
            torch.float16 if _np_dtype == np.float16 else torch.float32).numpy()
        _in_mm = np.memmap(os.path.join(_emb_dir, "input.dat"), dtype=_np_dtype, mode="w+", shape=_shape)
        _in_mm[:] = _fi_np
        _in_mm.flush()
        _out_mm = np.memmap(os.path.join(_emb_dir, "output.dat"), dtype=_np_dtype, mode="w+", shape=_shape)
        embedding_cache = {
            "input": torch.from_numpy(_in_mm),
            "output": torch.from_numpy(_out_mm),
        }
        del first_input, _fi_np
        utils.clean()
        print(json.dumps({"event": "emb_memmap", "shape": list(_shape), "dtype": str(_np_dtype),
                          "dir": _emb_dir,
                          "gb_each": round(float(np.prod(_shape)) * np.dtype(_np_dtype).itemsize / 1e9, 1)}),
              flush=True)
    else:
        _in_mm = _out_mm = None
        embedding_cache = {
            "input": first_input,
            "output": torch.zeros(first_input.shape, dtype=first_input.dtype, device=first_input.device)
        }

    # MADV_DONTFORK on the big host embedding buffers to prevent COW duplication
    # when TBE forks a compile worker. Without this, each layer's forward write
    # COW-copies ~43GB into the forked child, OOM-killing the 200GB container.
    if os.environ.get("HARNESS_EMB_DONTFORK", "1") != "0":
        import ctypes
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _PAGE = 4096
        _MADV_DONTFORK = 10
        for _name in ("input", "output"):
            _t = embedding_cache[_name]
            try:
                _addr = _t.data_ptr()
                _nbytes = _t.numel() * _t.element_size()
                _a = (_addr + _PAGE - 1) & ~(_PAGE - 1)
                _end = (_addr + _nbytes) & ~(_PAGE - 1)
                _len = max(0, _end - _a)
                if _len > 0:
                    _rc = _libc.madvise(ctypes.c_void_p(_a), ctypes.c_size_t(_len), _MADV_DONTFORK)
                    print(f"MADV_DONTFORK {_name}: {round(_len/1e9,1)}GB rc={_rc}", flush=True)
            except Exception as _e:
                print(f"MADV_DONTFORK {_name} failed: {_e}", flush=True)

    position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :] + torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int32)
    # Use embedding_cache["input"] (== first_input, or its memmap view) so this
    # works after first_input is freed in the memmap branch.
    _emb_in = embedding_cache["input"]
    if hasattr(model.model, "rotary_emb"):
        position_embeddings_cos, position_embeddings_sin = model.model.rotary_emb(_emb_in[0], position_ids)
    else:
        position_embeddings_cos, position_embeddings_sin = None, None

    attention_mask = _prepare_4d_causal_attention_mask(None, (args.batch_size, args.ctx_size), _emb_in[:args.batch_size], 0)

    # Device is auto-detected (npu→cuda→cpu), matching hessian_general_model.py.
    # This makes one code path serve both NPU and CUDA — the CUDA run scripts no
    # longer need to sed-patch this line. TORCH_DEVICE overrides for odd setups.
    if os.environ.get("TORCH_DEVICE"):
        device = torch.device(os.environ["TORCH_DEVICE"])
    else:
        device = torch.device("npu" if hasattr(torch, "npu") and torch.npu.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    _max_layers = int(os.environ.get("HARNESS_MAX_LAYERS", "0"))  # 0 = all; >0 caps for profiling
    for i in tqdm.tqdm(range(len(model.model.layers))):
        if _max_layers and i >= _max_layers:
            glog.info(f"[ft-profile] HARNESS_MAX_LAYERS={_max_layers} reached, stopping early")
            break
        utils.clean()

        def _cgroup_gb():
            # cgroup memory usage (the real container limit, ~192-300Gi), which
            # /proc/meminfo (host view) does NOT reflect. -1 if unavailable.
            for _p in ("/sys/fs/cgroup/memory.current",
                       "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
                try:
                    with open(_p) as _f:
                        return round(int(_f.read().strip()) / 1e9, 1)
                except Exception:
                    continue
            return -1

        if args.ft_epochs > 0:
            position_ids = position_ids.to(device)
            if position_embeddings_sin is not None:
                position_embeddings = (position_embeddings_cos.to(device), position_embeddings_sin.to(device))
            else:
                position_embeddings = None
            attention_mask = attention_mask.to(device)
            model.model.layers[i].to(device)
            _in_t = embedding_cache["input"]
            _out_t = embedding_cache["output"]
            print(json.dumps({"event": "fwd_start", "layer": i, "cgroup_gb": _cgroup_gb(),
                              "n_batches": args.devset_size // args.batch_size}), flush=True)
            for j in range(args.devset_size // args.batch_size):
                _sl = slice(args.batch_size * j, args.batch_size * (j + 1))
                if position_embeddings is None:
                    result = model.model.layers[i](
                        _in_t[_sl].to(device),
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_attentions=False
                    )
                else:
                    result = model.model.layers[i](
                        _in_t[_sl].to(device),
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_attentions=False
                    )
                # Copy D2H directly into the preallocated output slice. The old
                # `_out_t[_sl] = result[0].cpu()` minted a fresh glibc-malloc'd host
                # tensor every batch; `del` frees them but glibc retains the
                # fragmented arenas (malloc_trim only returns the heap top), so
                # cgroup RSS climbed ~36-43GB/layer → OOM. copy_ writes into
                # existing storage via a bounded, reused D2H staging buffer.
                # HARNESS_EMB_LEGACY_COPY=1 reverts to the synchronous pre-memfix
                # write (mints a fresh host tensor) to isolate whether copy_/del
                # degrades finetune (e.g. async D2H racing the `del result` free).
                if _emb_legacy:
                    _out_t[_sl] = result[0].cpu()
                else:
                    _out_t[_sl].copy_(result[0])
                del result
                # Periodically flush memmap (release dirty pages) + trim heap +
                # release NPU pinned-host pool so per-layer RSS stays bounded.
                if j % 256 == 255:
                    if _emb_memmap:
                        try:
                            _out_mm.flush()
                            _in_mm.flush()
                        except Exception:
                            pass
                    utils.clean()
            print(json.dumps({"event": "fwd_done", "layer": i, "cgroup_gb": _cgroup_gb()}), flush=True)
            model.model.layers[i].cpu()
            position_ids = position_ids.cpu()
            attention_mask = attention_mask.cpu()
            utils.clean()

        quantize_llama_layer(
            layer=model.model.layers[i],
            idx=i,
            cb=cb,
            args=args,
            device=device,
            pre_orig_emb=embedding_cache["input"],
            orig_emb=embedding_cache["output"],
            position_embeddings=position_embeddings,
            position_ids=position_ids
        )

        # Swap input/output for the next layer by REFERENCE (no clone). The old
        # clone() created a 3rd full [devset,ctx,hidden] tensor (~86GB → peak
        # ~258GB fp32) that, accumulating across layers, OOM-killed the container
        # at layer 1/2. Reference swap keeps only 2 buffers. The next layer's
        # forward overwrites the reused-as-output old-input buffer, already
        # consumed by layer i. Works for both in-RAM and memmap backings.
        # HARNESS_EMB_LEGACY_COPY=1 reverts to the pre-memfix clone (input gets a
        # fresh copy of output; output buffer untouched) to isolate whether the
        # reference swap degrades finetune.
        if _emb_legacy:
            embedding_cache['input'] = embedding_cache['output'].detach().clone()
        else:
            embedding_cache['input'], embedding_cache['output'] = \
                embedding_cache['output'], embedding_cache['input']


if __name__ == '__main__':
    glog.info("into function")
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    # batch_size is a device characteristic (HBM headroom), not a recipe value:
    # per-model scripts stay device-agnostic and let env pick it per device.
    if os.environ.get("HARNESS_Q_BATCH"):
        args.batch_size = int(os.environ["HARNESS_Q_BATCH"])
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
