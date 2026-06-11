# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""样例 13：守卫式 recovery-HYBRID 的 dense FFN / CNN 剪枝（部署式"实际应用"）。

演示要点：
 - 一次校准 → 在 recovery MENU {none, bias, ls} 上各剪一份副本；
 - 用独立小验证集实测择优（默认 top-1 保真度，无需标签）；
 - 仅当某 recovery 严格优于 fallback(none) 时切换 —— 守卫保证不劣于不补偿基线；
 - 校准规模决定 bias↔ls 的胜负，守卫自动选择；小校准要 test 侧稳健可传 recovery_menu=("none","ls")。
运行：python examples/pruning/recovery_hybrid.py
"""

from _models import make_mlp, mlp_data, make_cnn, cnn_data, count_params
from amct_pytorch.experimental.pruning import guarded_recovery_hybrid_prune


def main():
    # dense FFN
    model = make_mlp()
    calib, val = mlp_data(n=8), mlp_data(n=3, batch=2)
    before = count_params(model)
    res = guarded_recovery_hybrid_prune(
        model, data=calib, prune_ratio=0.5, domain="dense", eval_data=val, apply=True
    )
    _ = model(val[0])
    print("[13][dense]", res.summary())

    # CNN 通道
    cmodel = make_cnn()
    ccalib, cval = cnn_data(n=4), cnn_data(n=2)
    cbefore = count_params(cmodel)
    cres = guarded_recovery_hybrid_prune(
        cmodel, data=ccalib, prune_ratio=0.3, domain="cnn", eval_data=cval, apply=True
    )
    _ = cmodel(cval[0])
    print("[13][cnn]  ", cres.summary())

    # 守卫不变量：验证集上绝不劣于 fallback(none)
    assert res.gain_over_fallback >= 0.0 and cres.gain_over_fallback >= 0.0
    assert count_params(model) < before and count_params(cmodel) < cbefore


if __name__ == "__main__":
    main()
