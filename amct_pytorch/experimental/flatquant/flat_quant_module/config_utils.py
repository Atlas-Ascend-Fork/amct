
class FlatConfig():
    # activation
    a_bits = 4
    a_groupsize = -1
    a_sym = True

    # weight
    w_bits = 4
    w_groupsize = -1
    w_sym = True

    # KV-cache quantization
    use_kcache_quant = False
    use_vcache_quant = False
    k_bits = 16
    k_sym = False
    v_bits = 16
    v_sym = False

    # A special control for o_proj & down_proj
    use_o_quant = False
    use_down_quant = True

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
