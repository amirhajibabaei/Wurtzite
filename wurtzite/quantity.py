# +
"""
TODO:
    mass, distance, time, energy, velocity,
    force, torque, temperature, pressure,
    dynamic_viscosity, charge, dipole,
    electric_field, density
"""
from __future__ import annotations

import abc

from ase.calculators.lammps import convert

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
    def ase_unit(self) -> float:
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
        if unit is not None:
            unit = unit.lower()
        if unit is None or unit == self.default_unit:
            coef = 1.0
        elif unit == "ase":
            coef = self.ase_unit
        elif unit.startswith("lammps"):
            q = self.__class__.__name__.lower()
            u = unit.split("_")[1]
            coef = convert(1, q, u, "ASE") / self.ase_unit
        else:
            coef = self.derived_units[unit]
        return coef


class Time(Quantity):
    """
    Defined units:
        fs: femtosecond
        ps: picosecond
        ns: nanosecond
        s:  second

    Other unit systems:
        ase
        lammps_real
        lammps_metal
        etc.

    """

    @property
    def default_unit(self) -> str:
        return "fs"

    @property
    def ase_unit(self) -> float:
        return convert(1, "time", "real", "ASE")

    @property
    def derived_units(self) -> dict[str, float]:
        return {"ps": 1e3, "ns": 1e6, "s": 1e15}


def test_Time() -> bool:
    for u in "fs ps ns s".split():
        assert Time(1, u).get_value(u) == 1

    def are_close(a, b, rtol=1e-8):
        return abs(a - b) / min(abs(a), abs(b)) < rtol

    t = Time(1.0, "ase")
    assert are_close(t.get_value("fs"), t.get_value("lammps_real"))
    assert are_close(t.get_value("ps"), t.get_value("lammps_metal"))
    return True


if __name__ == "__main__":
    test_Time()
