import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    yield


@pytest.fixture
def cpu_device():
    return torch.device("cpu")
