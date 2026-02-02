''' 
Experimental predefined quant config; move to amct_pytorch/config once made official
'''

INT4_FLAT_QUANT_CFG = {
    'batch_num': 4,
    'quant_cfg': {
        'inputs': {
            'enable_quant': True,
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
        'flatquant_attn': {
            'use_kcache_quant': False,
            'k_bits': 16,
            'use_vcache_quant': False,
            'v_bits': 16,
            'use_o_quant': False
        },
        'flatquant_attn_spda': {},
        'flatquant_mlp': {}
    },
    'skip_layers': {'lm_head'}
}