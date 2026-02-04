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
            'use_kcache_quant': False,
            'k_bits': 16,
            'use_vcache_quant': False,
            'v_bits': 16,
            'use_o_quant': False
        },
    },
    'skip_layers': {'lm_head'}
}