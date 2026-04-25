from .base_solver import BaseSolver

class BlockwiseSolver(BaseSolver):
    def solve(self, model, calibration_data):
        if self.adapter is None:
            raise ValueError("Adapter not set. Call set_adapter() first.")
        quantizable_layers = self.adapter.get_quantizable_layers(model)
        for layer_name in quantizable_layers:
            self._quantize_layer(model, layer_name, calibration_data)
        return model

    def _quantize_layer(self, model, layer_name, calibration_data):
        pass