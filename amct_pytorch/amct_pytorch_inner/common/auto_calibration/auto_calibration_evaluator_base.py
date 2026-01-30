#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

model evaluator base class

"""


from typing import Tuple


class AutoCalibrationEvaluatorBase:
    """" the base class for ModelEvaluator"""
    def __init__(self):
        """
        Function:
                __init__ function of class
        """

    def calibration(self, model): # pylint: disable=R0201
        """
        Function:
                do the calibration
        Parameter:
                if framework is caffe:
                    model: the prototxt model define file of caffe model
                    weights: the binary caffemodel file of caffe model
                if framework is tensorflow:
                    model: the graph of model
                    outputs (list): a list of output nodes.
        Return:
                None
        """
        raise NotImplementedError

    def evaluate(self, model): # pylint: disable=R0201
        """
        Function:
                evaluate the input models, get the eval metric of model
        Parameter:
                if framwork is caffe:
                    model: the prototxt model define file of caffe model
                    weights: the binary caffemodel file of caffe model
                if framwork is tensorflow:
                    model: the graph of model
                    outputs (list): a list of output nodes.
        Return:
                metric: the eval metric of input caffe model, such as:
                top1 accuracy for classification model, mAP for
                detection model
        """
        raise NotImplementedError



    def metric_eval(self, original_metric, new_metric) -> Tuple[bool, float]: # pylint: disable=R0201
        """
        Function:
                whether the gap between new metric and original metric
                can satisfy the requirement.
        Parameter:
                original_metric: metric of eval original caffe model
                new_metric: metric of eval quantized caffe model
        Return:
                metric_eval (Tuple[bool, float]): A tuple of whether the
                metric is satisfied and loss function value.
        """
        raise NotImplementedError
