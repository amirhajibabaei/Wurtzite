# +
from __future__ import annotations

import abc

ValueType = float


class Quantity(abc.ABC):
    def __init__(self, value: ValueType, unit: str | None = None):
        """
        If unit == None, default unit is implied.
        """
        self._value = value * self._get_coef(unit)

    # For children:

    @property
    @abc.abstractmethod
    def default_unit(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def derived_units(self) -> dict[str, float]:
        ...

    # Derived:

    def get_value(self, unit: str | None = None) -> ValueType:
        """
        If unit == None, default unit is implied.
        """
        return self._value / self._get_coef(unit)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {self._value} {self.default_unit}"

    # Hidden:

    def _get_coef(self, unit: str | None) -> float:
        if unit is None or unit == self.default_unit:
            return 1
        else:
            return self.derived_units[unit]


class Time(Quantity):
    @property
    def default_unit(self) -> str:
        return "fs"

    @property
    def derived_units(self) -> dict[str, float]:
        return {"ps": 1e3, "ns": 1e6, "s": 1e15}


def test_Time() -> bool:
    for u in "fs ps ns s".split():
        assert Time(1, u).get_value(u) == 1
    return True


if __name__ == "__main__":
    test_Time()
