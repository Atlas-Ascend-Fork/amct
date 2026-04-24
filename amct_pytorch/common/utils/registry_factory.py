class RegistryItem:
    def __init__(self, name, item, force):
        self.name = name
        self.item = item
        self.force = force


class Registry:
    def __init__(self, name):
        self._name = name
        self._registry = {}

    def register(self, item, name=None, force=False):
        if name is None:
            name = item.__name__
        if name in self._registry and not force:
            raise KeyError(f"{name} is already registered in {self._name}")
        self._registry[name] = RegistryItem(name, item, force)
        return item

    def get(self, name):
        if name not in self._registry:
            raise KeyError(f"{name} not found in {self._name}")
        return self._registry[name].item

    def list_all(self):
        return list(self._registry.keys())

    def __contains__(self, name):
        return name in self._registry