# coding=utf-8
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

from collections import namedtuple
import yaml


LayerBits = namedtuple("LayerBits", ("w", "a"))

_BIT_KEYS = ("w_bits", "a_bits")
_ALLOWED_BITS = (4, 8, 16)
_LINEAR_GROUPS = ("attn-linear", "mlp", "moe")
_CACHE_GROUP = "attn-cache"


class _GroupBits:
    """Subscript proxy: ``bits["q_proj"]`` -> ``LayerBits(w, a)``."""

    __slots__ = ("_policy", "_group")

    def __init__(self, policy: "BitPolicy", group: str):
        self._policy = policy
        self._group = group

    def __getitem__(self, name: str) -> LayerBits:
        return LayerBits(*self._policy.linear_bits(name=name, group=self._group))

    @property
    def default(self) -> LayerBits:
        """Group-level (w, a) — falls through to top-level when the group has
        no complete entry of its own.
        """
        return LayerBits(*self._policy.linear_bits(group=self._group))


class BitPolicy:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        if ("w_bits" in self.cfg) != ("a_bits" in self.cfg):
            present = [k for k in _BIT_KEYS if k in self.cfg]
            raise ValueError(
                f"Incomplete bit entry at top level: must set both 'w_bits' and 'a_bits', "
                f"got {present}."
            )
        self.w_bits = int(self.cfg.get("w_bits", 16))
        self.a_bits = int(self.cfg.get("a_bits", 16))
        self._validate()

    def __getitem__(self, group: str) -> _GroupBits:
        return _GroupBits(self, group)

    @classmethod
    def from_yaml(cls, path: str) -> "BitPolicy":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"bit_config yaml at {path} must be a mapping at top level")
        _validate_bit_config(cfg)
        return cls(cfg)

    def cache_bits(self, key: str) -> int:
        cache = self.cfg.get(_CACHE_GROUP) or {}
        return int(cache.get(key, 16))

    def has_quant_cache(self) -> bool:
        cache = self.cfg.get(_CACHE_GROUP) or {}
        return any(int(v) < 16 for v in cache.values() if isinstance(v, int))

    def has_quant_linear(self) -> bool:
        if self.w_bits < 16 or self.a_bits < 16:
            return True
        for group in _LINEAR_GROUPS:
            sub = self.cfg.get(group)
            if isinstance(sub, dict) and _has_lt_16(sub):
                return True
        return False

    def linear_bits(self, name: str | None = None, group: str | None = None) -> tuple[int, int]:
        """Resolve (w_bits, a_bits) along ``group`` (dotted) / ``name``.

        Walks from most-specific (leaf name) to least-specific (top group),
        returning the first node that has a complete bit entry. Falls back to
        the top-level globals when nothing matches.
        """
        parts: list[str] = []
        if group:
            parts.extend(group.split("."))
        if name:
            parts.append(name)

        cursor = self.cfg
        chain: list[dict] = []
        for part in parts:
            sub = cursor.get(part) if isinstance(cursor, dict) else None
            if not isinstance(sub, dict):
                break
            chain.append(sub)
            cursor = sub

        for node in reversed(chain):
            if "w_bits" in node and "a_bits" in node:
                return int(node["w_bits"]), int(node["a_bits"])
        return self.w_bits, self.a_bits

    def summary(self) -> str:
        return "BitPolicy:\n" + yaml.safe_dump(self.cfg, sort_keys=False, default_flow_style=False).rstrip()

    def _validate(self):
        """Enforce: any nested entry that mentions one of w_bits/a_bits must
        mention both. (Top-level is exempt — w_bits/a_bits there are
        independent globals.)"""
        for group in _LINEAR_GROUPS:
            node = self.cfg.get(group)
            if isinstance(node, dict):
                _check_complete(node, group)


def ensure_bit_policy(args) -> BitPolicy:
    policy = getattr(args, "bit_policy", None)
    if policy is not None:
        return policy

    cfg = {
        "w_bits": int(getattr(args, "w_bits", 16)),
        "a_bits": int(getattr(args, "a_bits", 16)),
        _CACHE_GROUP: {
            "q": int(getattr(args, "q_bits", 16)),
            "k": int(getattr(args, "k_bits", 16)),
            "p": int(getattr(args, "p_bits", 16)),
            "v": int(getattr(args, "v_bits", 16)),
        },
    }
    policy = BitPolicy(cfg)
    setattr(args, "bit_policy", policy)
    return policy


def _check_complete(node: dict, path: str):
    has_w = "w_bits" in node
    has_a = "a_bits" in node
    if has_w != has_a:
        present = [k for k in _BIT_KEYS if k in node]
        raise ValueError(
            f"Incomplete bit entry at {path!r}: must set both 'w_bits' and 'a_bits', "
            f"got {present}."
        )
    for k, v in node.items():
        if k in _BIT_KEYS:
            continue
        if isinstance(v, dict):
            _check_complete(v, f"{path}.{k}")


def _validate_bit_value(path: str, value):
    if type(value) is not int:
        raise ValueError(f"{path} must be int, but got {type(value).__name__}.")
    if value not in _ALLOWED_BITS:
        raise ValueError(f"{path} must be one of {_ALLOWED_BITS}, but got {value}.")


def _validate_bit_config(node: dict, path: str = ""):
    for k, v in node.items():
        cur_path = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            _validate_bit_config(v, cur_path)
        else:
            _validate_bit_value(cur_path, v)


def _has_lt_16(node: dict) -> bool:
    for v in node.values():
        if isinstance(v, dict):
            if _has_lt_16(v):
                return True
        elif isinstance(v, int) and v < 16:
            return True
    return False
