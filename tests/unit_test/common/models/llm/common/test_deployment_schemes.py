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
    PLATFORM_QUANT_DTYPES,
    QUANT_SCHEMES,
    QuantSchemeKey,
    get_quant_scheme,
    resolve_deploy_platform,
    validate_platform_quant_scheme,
    validate_selection_quant_scheme,
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
        QuantSchemeKey("hifp", 8, 8),
    ],
)
def test_unsupported_ascend_scheme_is_rejected(key):
    with pytest.raises(ValueError, match="Unsupported deployment quantization scheme"):
        get_quant_scheme(key)


@pytest.mark.parametrize(
    "platform, quant_dtype",
    [
        ("A3", "int"),
        ("A3", "mxfp"),
        ("ascend950", "int"),
        ("ascend950", "mxfp"),
    ],
)
def test_platform_quant_dtype_cross_product(platform, quant_dtype):
    selection = SimpleNamespace(
        quant_layers=(
            SimpleNamespace(
                quant_spec=SimpleNamespace(
                    quant_dtype=quant_dtype, weight_bits=8, activation_bits=8
                )
            ),
        )
    )

    if quant_dtype in PLATFORM_QUANT_DTYPES[platform]:
        validate_platform_quant_scheme(selection, platform)
        return

    expected_platforms = sorted(
        candidate
        for candidate, dtypes in PLATFORM_QUANT_DTYPES.items()
        if candidate != platform and quant_dtype in dtypes
    )
    expected_hint = (
        f" Use --deploy_platform {' or '.join(expected_platforms)}."
        if expected_platforms
        else ""
    )
    with pytest.raises(ValueError) as exc_info:
        validate_platform_quant_scheme(selection, platform)
    assert str(exc_info.value) == (
        f"Deployment platform {platform} does not support quant_dtype "
        f"{quant_dtype!r}; supported values: {sorted(PLATFORM_QUANT_DTYPES[platform])}."
        f"{expected_hint}"
    )


def test_platform_rejects_unknown_platform():
    selection = SimpleNamespace(
        quant_layers=(
            SimpleNamespace(
                quant_spec=SimpleNamespace(
                    quant_dtype="int", weight_bits=8, activation_bits=8
                )
            ),
        )
    )

    with pytest.raises(ValueError, match="Unsupported Deployment platform"):
        validate_platform_quant_scheme(selection, "ascend960")


def test_selection_rejects_mixed_schemes():
    selection = SimpleNamespace(
        quant_layers=(
            SimpleNamespace(
                quant_spec=SimpleNamespace(
                    quant_dtype="int", weight_bits=8, activation_bits=8
                )
            ),
            SimpleNamespace(
                quant_spec=SimpleNamespace(
                    quant_dtype="mxfp", weight_bits=8, activation_bits=8
                )
            ),
        )
    )

    with pytest.raises(ValueError, match="one quantization scheme"):
        validate_selection_quant_scheme(selection)


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
