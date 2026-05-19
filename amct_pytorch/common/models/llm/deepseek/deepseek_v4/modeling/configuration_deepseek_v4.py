from transformers.configuration_utils import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    """Bilingual config: accepts both the canonical HuggingFace field names used
    in the official `config.json` (`hidden_size`, `num_hidden_layers`, ...) and
    the `ModelArgs`-style names that `Block / Attention / MoE / ...` actually
    read at runtime (`dim`, `n_layers`, ...). HF aliases win when both are
    provided so that loading the official config gives the right values.
    """
    model_type = "deepseek_v4"

    def __init__(
        self,
        # ---- ModelArgs-style canonical fields (also defaults) ----
        max_batch_size=4,
        max_seq_len=4096,
        dtype="fp8",
        scale_fmt="ue8m0",
        expert_dtype=None,
        scale_dtype="fp8",
        vocab_size=129280,
        dim=4096,
        moe_inter_dim=4096,
        n_layers=7,
        n_hash_layers=0,
        n_mtp_layers=1,
        n_heads=64,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        score_func="sqrtsoftplus",
        route_scale=1.0,
        swiglu_limit=0.0,
        q_lora_rank=1024,
        head_dim=512,
        rope_head_dim=64,
        norm_eps=1e-6,
        o_groups=8,
        o_lora_rank=1024,
        window_size=128,
        compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0),
        compress_rope_theta=40000.0,
        original_seq_len=0,
        rope_theta=10000.0,
        rope_factor=40,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        tie_word_embeddings=False,
        # ---- HF-style aliases (override ModelArgs-style when provided) ----
        hidden_size=None,
        num_hidden_layers=None,
        num_attention_heads=None,
        moe_intermediate_size=None,
        num_experts_per_tok=None,
        qk_rope_head_dim=None,
        rms_norm_eps=None,
        sliding_window=None,
        scoring_func=None,
        routed_scaling_factor=None,
        num_hash_layers=None,
        num_nextn_predict_layers=None,
        max_position_embeddings=None,
        # ---- Nested HF dicts we unpack ----
        rope_scaling=None,
        quantization_config=None,
        **kwargs,
    ):
        # 1) HF flat aliases -> ModelArgs names
        if hidden_size is not None:
            dim = hidden_size
        if num_hidden_layers is not None:
            n_layers = num_hidden_layers
        if num_attention_heads is not None:
            n_heads = num_attention_heads
        if moe_intermediate_size is not None:
            moe_inter_dim = moe_intermediate_size
        if num_experts_per_tok is not None:
            n_activated_experts = num_experts_per_tok
        if qk_rope_head_dim is not None:
            rope_head_dim = qk_rope_head_dim
        if rms_norm_eps is not None:
            norm_eps = rms_norm_eps
        if sliding_window is not None:
            window_size = sliding_window
        if scoring_func is not None:
            score_func = scoring_func
        if routed_scaling_factor is not None:
            route_scale = routed_scaling_factor
        if num_hash_layers is not None:
            n_hash_layers = num_hash_layers
        if num_nextn_predict_layers is not None:
            n_mtp_layers = num_nextn_predict_layers
        if max_position_embeddings is not None:
            max_seq_len = max_position_embeddings

        # 2) rope_scaling dict -> flat YaRN fields
        if rope_scaling is not None:
            if "factor" in rope_scaling:
                rope_factor = rope_scaling["factor"]
            if "original_max_position_embeddings" in rope_scaling:
                original_seq_len = rope_scaling["original_max_position_embeddings"]
            if "beta_fast" in rope_scaling:
                beta_fast = rope_scaling["beta_fast"]
            if "beta_slow" in rope_scaling:
                beta_slow = rope_scaling["beta_slow"]

        # 3) quantization_config dict -> dtype / scale_fmt for the runtime globals
        if quantization_config is not None:
            if quantization_config.get("quant_method") == "fp8":
                dtype = "fp8"
            if "scale_fmt" in quantization_config:
                scale_fmt = quantization_config["scale_fmt"]

        # 4) Bind ModelArgs-style fields (what Block / Attention / MoE read)
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.scale_fmt = scale_fmt
        self.expert_dtype = expert_dtype
        self.scale_dtype = scale_dtype
        self.vocab_size = vocab_size
        self.dim = dim
        self.moe_inter_dim = moe_inter_dim
        self.n_layers = n_layers
        self.n_hash_layers = n_hash_layers
        self.n_mtp_layers = n_mtp_layers
        self.n_heads = n_heads
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts
        self.score_func = score_func
        self.route_scale = route_scale
        self.swiglu_limit = swiglu_limit
        self.q_lora_rank = q_lora_rank
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.norm_eps = norm_eps
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.window_size = window_size
        self.compress_ratios = tuple(compress_ratios)
        self.compress_rope_theta = compress_rope_theta
        self.original_seq_len = original_seq_len
        self.rope_theta = rope_theta
        self.rope_factor = rope_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        # 5) Keep HF-style aliases on self too — HF utilities (generation,
        # tokenizer max-length checks, transformers internals) read these
        # directly, so we mirror every translated field, not just the obvious
        # ones, to avoid AttributeError surprises later.
        self.hidden_size = dim
        self.num_hidden_layers = n_layers
        self.num_attention_heads = n_heads
        self.moe_intermediate_size = moe_inter_dim
        self.num_experts_per_tok = n_activated_experts
        self.qk_rope_head_dim = rope_head_dim
        self.rms_norm_eps = norm_eps
        self.sliding_window = window_size
        self.scoring_func = score_func
        self.routed_scaling_factor = route_scale
        self.num_hash_layers = n_hash_layers
        self.num_nextn_predict_layers = n_mtp_layers
        self.max_position_embeddings = max_seq_len
        self.rope_scaling = rope_scaling
        self.quantization_config = quantization_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        # Modern HF aliases `dtype` to `torch_dtype` at the same attribute slot,
        # so super() will overwrite our architectural `self.dtype = "fp8"` with
        # `torch.bfloat16` (from the official config's `torch_dtype: "bfloat16"`).
        # Keep HF semantics for `self.dtype / self.torch_dtype` (runtime load
        # dtype), and expose our v4 architectural quant-format selector under a
        # non-colliding name. Modeling code reads `config.arch_dtype`.
        self.arch_dtype = dtype
