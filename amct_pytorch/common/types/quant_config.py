from typing import Dict, Any, Optional


class QuantConfig:
    def __init__(
        self,
        algorithm: str = "minmax",
        weight_bits: int = 8,
        activation_bits: int = 8,
        quant_dtype: str = "int",
        group_size: Optional[int] = None,
        **kwargs
    ):
        self.algorithm = algorithm
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.quant_dtype = quant_dtype
        self.group_size = group_size
        self.extra_config = kwargs

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QuantConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "quant_dtype": self.quant_dtype,
            "group_size": self.group_size,
            **self.extra_config
        }