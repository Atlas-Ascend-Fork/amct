from .base_solver import BaseSolver

class GlobalSolver(BaseSolver):
    def solve(self, model, calibration_data):
        self._quantize_model(model, calibration_data)
        return model

    def _quantize_model(self, model, calibration_data):
        pass