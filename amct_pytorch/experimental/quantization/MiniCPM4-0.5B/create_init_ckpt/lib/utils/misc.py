import gc
import pdb
import sys

import glog
import torch
from tqdm import tqdm


def clean(do_gc=True):
    # gc.collect() walks every live Python object and is expensive on the 3B
    # stage2 hot path (called many times/linear inside the tune loop). Refcounting
    # already frees the per-iter tensors immediately after their `del`, so the
    # full collection is only needed to break reference cycles — fold it into the
    # per-layer clean and let hot callers pass do_gc=False.
    if do_gc:
        gc.collect()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
        # Release cached HOST (pinned) memory. The NPU host caching allocator
        # keeps a pinned staging block per H2D/D2H transfer, each tied to a
        # transfer event; a block can't be reused until its event completes, so
        # stage2's per-batch transfers/layer inflate the pool to ~one full
        # embedding (~43GB/layer). torch.npu.empty_cache() frees only DEVICE
        # cache, not these — cgroup anon then climbs ~43GB/layer (main-proc VmRSS
        # stays flat, since pinned host memory is charged to the cgroup but not
        # the python RSS) and OOM-kills the container. synchronize() lets in-flight
        # events complete so the blocks become unoccupied, then
        # _npu_hostEmptyCache() returns them to the OS.
        if do_gc:
            try:
                import torch_npu
                torch.npu.synchronize()
                torch_npu._C._npu_hostEmptyCache()
            except Exception:
                pass
    else:
        torch.cuda.empty_cache()
    # Release freed heap arenas back to the OS. glibc malloc keeps them by default,
    # which inflates cgroup RSS ~36GB/layer from stage2's transient result.cpu()
    # tensors and OOM-kills the container. malloc_trim returns the arenas so RSS
    # stays bounded. Only on the per-layer (do_gc) path to avoid syscall overhead
    # in the tune hot loop.
    if do_gc:
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def show_metrics(hatW, W_orig, H, msg):
    err_frob = (hatW - W_orig).square().sum() / W_orig.square().sum()
    err_proxy = (((hatW - W_orig) @ H) * (hatW - W_orig)).sum() / ((W_orig @ H) * W_orig).sum()
    glog.info(f"{msg} frob  error: {err_frob}")
    glog.info(f"{msg} proxy error: {err_proxy}")


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    use as:
    from lib import utils
    utils.ForkedPdb().set_trace()
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


'''


    for i in tqdm(range(args.devset_size // args.batch_size), desc=f"calculating dev emb for layer {idx}"):
        dev_emb[(args.batch_size * i):(args.batch_size * (i + 1)), :, :] = transformer_layer(
            dev_emb[(args.batch_size * i):(args.batch_size * (i + 1)), :, :].to(device),
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False)[0].cpu()

    act_error = (dev_emb.to(dtype_) - orig_emb.to(dtype_)).square().sum() / (orig_emb.to(dtype_) -
                                                                             orig_emb.to(dtype_).mean(
                                                                                 (0, 1))).square().sum()
    glog.info(f"layer {idx} act error: {act_error}")

    glog.info(f"saving activations for layer {idx}")

    torch.save(
        {
            'devset': devset,
            'dev_emb': dev_emb,
            'orig_emb': orig_emb,
            'after_layer': idx,
            'timestamp': str(datetime.datetime.now())
        }, f"{args.save_path}/dev_activations.new.pt")
    if os.path.isfile(f"{args.save_path}/dev_activations.pt"):
        os.remove(f"{args.save_path}/dev_activations.pt")
    os.rename(f"{args.save_path}/dev_activations.new.pt", f"{args.save_path}/dev_activations.pt")

    glog.info(f'"done processing layer {idx}, total time {time.time() - st_time} seconds')


'''
