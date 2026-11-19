# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest

from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
    validate_deploy_cli_args,
)
from amct_pytorch.common.models.llm.common.deployment.schemes import (
    QUANT_SCHEMES,
    QuantSchemeKey,
    get_quant_scheme,
    resolve_deploy_platform,
)


def test_deployment_modules_are_available_from_new_path():
    from amct_pytorch.common.models.llm.common.deployment.deploy_export import (
        make_description_from_plan as description,
    )
    from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
        QuantSpec,
    )

    assert callable(description)
    assert QuantSpec.__name__ == "QuantSpec"


def test_missing_deploy_platform_defaults_to_a3():
    assert resolve_deploy_platform(None) == "A3"
    assert resolve_deploy_platform("") == "A3"
    assert resolve_deploy_platform("ascend950") == "ascend950"


@pytest.mark.parametrize(
    "key, model_quant_type",
    [
        (QuantSchemeKey("int", 8, 8), "W8A8_DYNAMIC"),
        (QuantSchemeKey("mxfp", 8, 8), "W8A8_MXFP8"),
        (QuantSchemeKey("hifp", 8, 8), "W8A8_HIFP8"),
    ],
)
def test_supported_ascend_scheme_has_vllm_metadata(key, model_quant_type):
    scheme = get_quant_scheme(key)

    assert scheme is QUANT_SCHEMES[key]
    assert scheme.model_quant_type == model_quant_type


@pytest.mark.parametrize(
    "key",
    [
        QuantSchemeKey("fp", 8, 8),
        QuantSchemeKey("int", 4, 8),
        QuantSchemeKey("hifp", 4, 8),
    ],
)
def test_unsupported_ascend_scheme_is_rejected(key):
    with pytest.raises(ValueError, match="Unsupported deployment quantization scheme"):
        get_quant_scheme(key)


def test_cli_validator_requires_tensor_json_file(tmp_path):
    args = SimpleNamespace(
        deploy_format="ascend",
        granularity="tensor",
        quant_layers_config=str(tmp_path / "missing.json"),
    )

    with pytest.raises(ValueError, match="existing file"):
        validate_deploy_cli_args(args)


def test_cli_validator_does_not_require_global_quant_dtype(tmp_path):
    path = tmp_path / "layers.json"
    path.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        deploy_format="ascend",
        granularity="tensor",
        quant_layers_config=str(path),
        quant_dtype="",
    )

    validate_deploy_cli_args(args)
