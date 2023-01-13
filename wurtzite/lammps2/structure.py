# +
from __future__ import annotations

import abc
from typing import Sequence

import numpy as np
from ase.calculators.lammps import convert
from lammps import lammps

import wurtzite.lammps2._backend as backend
from wurtzite.atomic_structure import AtomicStructure, DynamicStructure
from wurtzite.lammps2.forcefield import ForceField


class LAMMPS(DynamicStructure):
    _default_units = "real"
    _lmp: lammps

    @abc.abstractmethod
    def get_units(self) -> str:
        ...

    @abc.abstractmethod
    def get_types(self) -> dict[str, int]:
        ...

    def apply_forcefield_(self, ff: ForceField) -> None:
        self._lmp.commands_list(ff.get_commands(self.get_units(), self.get_types()))

    def get_potential_energy(self, units: str = "ASE") -> float:
        self._lmp.command("run 0")
        e = self._lmp.get_thermo("pe")
        e = convert(e, "energy", self.get_units(), units)
        return e


class FullStyle(LAMMPS):
    def __init__(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        symbols: Sequence[str],
        pbc: bool | tuple[bool, bool, bool],
        units: str | None = None,
    ):
        if units is None:
            units = self._default_units
        self._lmp, self._types = backend._create_lammps(
            positions, cell, symbols, pbc, units
        )
        self._positions = positions
        self._cell = cell
        self._symbols = symbols
        self._pbc = backend._pbc_tuple(pbc)
        self._units = units

    # From parents:

    def get_units(self) -> str:
        return self._units

    def get_types(self) -> dict[str, int]:
        return self._types

    def get_pbc(self) -> tuple[bool, bool, bool]:
        return self._pbc

    def get_cell(self) -> np.ndarray:
        return self._cell

    def get_positions(self) -> np.ndarray:
        return self._cell

    def get_chemical_symbols(self) -> Sequence[str]:
        return self._symbols

    def set_positions(self, positions: np.ndarray) -> None:
        backend._update_lammps(self._lmp, self._pbc, self._cell, positions, self._units)
        self._positions = positions

    def set_cell(self, cell: np.ndarray) -> None:
        backend._update_lammps_cell(self._lmp, cell, self._units)
        self._cell = cell

    def set_pbc(self, pbc: bool | tuple[bool, bool, bool]) -> None:
        pbc = backend._pbc_tuple(pbc)
        backend._update_lammps_pbc(self._lmp, pbc)
        self._pbc = pbc

    # Utility:

    @staticmethod
    def from_atomic_structure(
        struc: AtomicStructure, units: str | None = None
    ) -> FullStyle:
        return FullStyle(
            struc.get_positions(),
            struc.get_cell(),
            struc.get_chemical_symbols(),
            struc.get_pbc(),
        )
