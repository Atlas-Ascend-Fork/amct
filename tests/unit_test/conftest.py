import shutil
import random
from pathlib import Path

import numpy as np
import pytest
import torch

SAFETENSORS_TMP_DIR = Path(__file__).resolve().parent / "tmp"


@pytest.fixture(autouse=True, scope="session")
def _safetensors_tmp_cleanup():
    """Create and tear down the shared safetensors mock directory."""
    SAFETENSORS_TMP_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if SAFETENSORS_TMP_DIR.exists():
        shutil.rmtree(SAFETENSORS_TMP_DIR)


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
