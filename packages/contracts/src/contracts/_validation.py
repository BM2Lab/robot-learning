"""Standard-library-only helpers for V0 contract validation."""
from dataclasses import asdict
import json
import math
from numbers import Real


def text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def integer(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def names(values, field):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of names")
    result = tuple(text(v, field) for v in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{field} must be nonempty and unique")
    return result


def number(value, field, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite numeric data")
    if positive and value <= 0:
        raise ValueError(f"{field} must be positive")
    return float(value)


class JsonContract:
    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    @classmethod
    def from_json(cls, value):
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("Contract JSON must be an object")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)
