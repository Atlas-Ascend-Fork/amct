import pytest

from amct_pytorch.common.utils.check_params import (
    check_parameters_in_schema,
    check_params,
)


@check_params(int, str)
def _typed_fn(a, b, c=0.5):
    return a, b, c


def test_check_params_passes_when_types_match():
    assert _typed_fn(1, "ok") == (1, "ok", 0.5)


def test_check_params_rejects_wrong_positional_type():
    with pytest.raises(TypeError, match="argument a must be"):
        _typed_fn("not-an-int", "ok")


def test_check_params_rejects_wrong_keyword_type():
    with pytest.raises(TypeError, match="argument b must be"):
        _typed_fn(1, b=42)


def test_check_params_ignores_unchecked_args():
    # `c` is unchecked, any value should pass.
    assert _typed_fn(1, "ok", c="anything") == (1, "ok", "anything")


def test_check_params_accepts_class_via_issubclass_branch():
    @check_params(Exception)
    def fn(exc_cls):
        return exc_cls

    assert fn(ValueError) is ValueError


def test_check_params_rejects_wrong_class_via_issubclass_branch():
    @check_params(Exception)
    def fn(exc_cls):
        return exc_cls

    with pytest.raises(TypeError):
        fn(int)  # int is a class but not a subclass of Exception


class _SchemaArg:
    def __init__(self, name):
        self.name = name


class _Schema:
    def __init__(self, names):
        self.arguments = [_SchemaArg(n) for n in names]


def _make_func_with_schemas(arg_names):
    def fn():
        pass

    fn._schemas = {"v1": _Schema(arg_names)}
    return fn


def test_check_parameters_in_schema_finds_all_when_present():
    fn = _make_func_with_schemas(["x", "y", "z"])
    assert check_parameters_in_schema(fn, "x", "y") is True


def test_check_parameters_in_schema_returns_false_when_missing():
    fn = _make_func_with_schemas(["x", "y"])
    assert check_parameters_in_schema(fn, "x", "missing") is False


def test_check_parameters_in_schema_accepts_list_argument():
    fn = _make_func_with_schemas(["x", "y", "z"])
    assert check_parameters_in_schema(fn, ["x", "z"]) is True
    assert check_parameters_in_schema(fn, ["x", "absent"]) is False
