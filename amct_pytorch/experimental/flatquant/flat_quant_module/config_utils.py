from enum import Enum


class InnerConfig(Enum):
    # activation
    a_bits = 4
    a_groupsize = -1
    a_asym = False

    # weight
    w_bits = 4
    w_groupsize = -1
    w_asym = False

    # KV-cache quantization
    use_kcache_quant = False
    use_vcache_quant = False
    k_bits = 16
    k_asym = True
    v_bits = 16
    v_asym = True

    # A special control for o_trans and o_proj (different from other proj)
    use_o_quant = False

    # Other quantization parameters
    add_diag = True
    lwc = True
    lac = False
    diag_alpha = 0.3

    # Calibration
    epochs = 15
    nsamples = 128
    cali_bsz = 4
    flat_lr = 5e-3
    cali_trans = True
    