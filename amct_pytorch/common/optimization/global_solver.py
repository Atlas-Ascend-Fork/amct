@SOLVER_REGISTRY.register(name="global", description="Global calibration solver")
class GlobalSolver(BaseSolver):
    def __init__(
            self,
            model: nn.Module,
            optimizer_fn: Callable,
            lr_scheduler_fn: Callable = None,
            block_size: int = 128,
            max_iters: int = 100,
    ):
        super().__init__(model, None, None, max_iters)
        self.granularity = "model"