''' 
Experimental predefined quant config; move to amct_pytorch/config once made official
'''

INT4_FLAT_QUANT_CFG = {
    'quant_cfg': {
        'inputs': {
            'type': 'int4',
            'symmetric': True,
            'strategy': 'token'
        },
        'weights': {
            'type': 'int4',
            'symmetric': True,
            'strategy': 'channel',
        },
    },
    'algorithm': {
        'flatquant': {
            # KV-cache quantization
            'use_kcache_quant': False,
            'k_bits': 16,
            'k_sym': False,
            'use_vcache_quant': False,
            'v_bits': 16,
            'v_sym': False,

            # A special control for o_proj & down_proj
            'use_o_quant': False,
            'use_down_quant': True,

            # Other quantization parameters
            'add_diag': True,
            'lac': False,
            'lwc': True,
            'diag_alpha': 0.3,

            # Calibration
            'epochs': 15,
            'cali_bsz': 4,
            'flat_lr': 5e-3,
            'cali_trans': True,
        },
    },
    'skip_layers': {'lm_head'}
}