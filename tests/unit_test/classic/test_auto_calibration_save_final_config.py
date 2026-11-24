#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
# ----------------------------------------------------------------------------

import json

from amct_pytorch.classic.graph_based.amct_pytorch.common.auto_calibration.accuracy_based_auto_calibration_base import (
    AccuracyBasedAutoCalibrationBase,
)


def _build_calibration(tmp_path, config, final_config, **extra):
    save_dir = tmp_path / "out" / "model"
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    config_file = tmp_path / "quant_config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")

    calibration = object.__new__(AccuracyBasedAutoCalibrationBase)
    calibration.config_file = str(config_file)
    calibration.save_dir = str(save_dir)
    calibration.final_config = final_config
    for name, value in extra.items():
        setattr(calibration, name, value)

    result_file = save_dir.parent / "accuracy_based_auto_calibration_final_config.json"
    return calibration, result_file


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class _Node:
    def __init__(self, node_type):
        self.type = node_type


class _Graph:
    def __init__(self, nodes=None):
        self._nodes = nodes or {}

    def get_node_by_name(self, name):
        if name not in self._nodes:
            raise RuntimeError('Cannot find node "{}" in graph'.format(name))
        return self._nodes[name]


def test_save_final_config_handles_anonymous_node_without_original_graph(tmp_path):
    calibration, result_file = _build_calibration(
        tmp_path,
        config={"conv::common": {"quant_enable": True, "w_bits": 8}},
        final_config={"anon_layer": False},
    )

    calibration.save_final_config()

    result = _read_json(result_file)
    assert result["anon_layer"]["quant_enable"] is False


def test_save_final_config_falls_back_to_original_quant_config(tmp_path):
    original_quant_config = {"anon_layer": {"quant_enable": True, "w_bits": 4}}
    calibration, result_file = _build_calibration(
        tmp_path,
        config={},
        final_config={"anon_layer": False},
        original_quant_config=original_quant_config,
    )

    calibration.save_final_config()

    result = _read_json(result_file)
    assert result["anon_layer"] == {"quant_enable": False, "w_bits": 4}
    assert calibration.original_quant_config["anon_layer"]["quant_enable"] is True


def test_save_final_config_falls_back_when_node_absent_from_graph(tmp_path):
    calibration, result_file = _build_calibration(
        tmp_path,
        config={},
        final_config={"anon_layer": False},
        original_graph=_Graph(),
        original_quant_config={"anon_layer": {"quant_enable": True, "w_bits": 4}},
    )

    calibration.save_final_config()

    result = _read_json(result_file)
    assert result["anon_layer"] == {"quant_enable": False, "w_bits": 4}


def test_save_final_config_deepcopies_template_for_each_anonymous_node(tmp_path):
    calibration, result_file = _build_calibration(
        tmp_path,
        config={"conv::common": {"quant_enable": True, "w_bits": 8}},
        final_config={"anon_a": False, "anon_b": True},
        original_graph=_Graph({"anon_a": _Node("conv"), "anon_b": _Node("conv")}),
    )

    calibration.save_final_config()

    result = _read_json(result_file)
    assert result["anon_a"]["quant_enable"] is False
    assert result["anon_b"]["quant_enable"] is True
    assert result["conv::common"]["quant_enable"] is True
