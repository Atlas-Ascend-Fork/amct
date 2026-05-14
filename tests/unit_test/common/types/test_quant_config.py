from amct_pytorch.common.types.quant_config import QuantConfig


def test_default_values():
    cfg = QuantConfig()
    assert cfg.algorithm == "minmax"
    assert cfg.weight_bits == 8
    assert cfg.activation_bits == 8
    assert cfg.quant_dtype == "int"
    assert cfg.group_size is None
    assert not cfg.extra_config


def test_explicit_values_and_extra_kwargs_collected():
    cfg = QuantConfig(
        algorithm="awq",
        weight_bits=4,
        activation_bits=16,
        quant_dtype="mxfp",
        group_size=128,
        custom_field="x",
        another=1,
    )
    assert cfg.algorithm == "awq"
    assert cfg.weight_bits == 4
    assert cfg.activation_bits == 16
    assert cfg.quant_dtype == "mxfp"
    assert cfg.group_size == 128
    assert cfg.extra_config == {"custom_field": "x", "another": 1}


def test_to_dict_includes_known_and_extra_fields():
    cfg = QuantConfig(weight_bits=4, custom="z")
    d = cfg.to_dict()
    assert d["weight_bits"] == 4
    assert d["custom"] == "z"
    assert set(d) >= {
        "algorithm",
        "weight_bits",
        "activation_bits",
        "quant_dtype",
        "group_size",
        "custom",
    }


def test_from_dict_round_trip_preserves_known_fields():
    src = QuantConfig(algorithm="awq", weight_bits=4, group_size=64, custom="z")
    rebuilt = QuantConfig.from_dict(src.to_dict())
    assert rebuilt.algorithm == src.algorithm
    assert rebuilt.weight_bits == src.weight_bits
    assert rebuilt.group_size == src.group_size
    assert rebuilt.extra_config == src.extra_config
