from typing import Optional
from .base_adapter import BaseModelAdapter
from ..utils.registry_factory import Registry

ClassicAdapterRegistry = Registry("classic_adapter")

def register_adapter(adapter_cls, name: str, adapter_type: str = "classic"):
    if adapter_type == "classic":
        ClassicAdapterRegistry.register(adapter_cls, name=name)

def get_adapter(model_name: str, adapter_type: Optional[str] = None):
    if adapter_type == "classic" or adapter_type is None:
        if model_name in ClassicAdapterRegistry.list_all():
            return ClassicAdapterRegistry.get(model_name)
    raise ValueError(f"Unknown model: {model_name}")

class ModelRegistry:
    @staticmethod
    def register_classic(adapter_cls, name: str):
        register_adapter(adapter_cls, name, "classic")

    @staticmethod
    def get(model_name: str, adapter_type: Optional[str] = None):
        return get_adapter(model_name, adapter_type)