import pytest

from amct_pytorch.algorithms.register_algo import Algorithm


class _Op:
    pass


class _OpA(_Op):
    pass


class _OpB(_Op):
    pass


class _OpC(_Op):
    pass


def test_initial_state_is_empty():
    a = Algorithm()
    assert a.algo == {}
    assert a.quant_to_deploy == {}
    assert a.quant_op == []


def test_register_records_src_to_quant_mapping():
    a = Algorithm()
    a.register("alg", "Linear", _OpA, _OpB)
    assert a.algo == {"alg": {"Linear": _OpA}}


def test_multiple_src_ops_share_algorithm_name():
    a = Algorithm()
    a.register("alg", "Linear", _OpA, _OpB)
    a.register("alg", "Conv2d", _OpC, _OpB)
    assert a.algo["alg"] == {"Linear": _OpA, "Conv2d": _OpC}


def test_register_scalar_deploy_op_appended_to_list():
    a = Algorithm()
    a.register("alg", "Linear", _OpA, _OpB)
    assert a.quant_to_deploy[_OpA] == [_OpB]


def test_register_list_deploy_op_extends_existing_list():
    a = Algorithm()
    a.register("alg", "Linear", _OpA, [_OpB, _OpC])
    assert set(a.quant_to_deploy[_OpA]) == {_OpB, _OpC}


def test_quant_to_deploy_is_deduplicated():
    a = Algorithm()
    a.register("alg", "Linear", _OpA, _OpB)
    a.register("alg", "Linear", _OpA, _OpB)
    assert a.quant_to_deploy[_OpA] == [_OpB]


def test_repeated_registration_can_extend_deploy_with_new_target():
    a = Algorithm()
    a.register("alg", "Linear", _OpA, _OpB)
    a.register("alg", "Linear", _OpA, _OpC)
    assert set(a.quant_to_deploy[_OpA]) == {_OpB, _OpC}
