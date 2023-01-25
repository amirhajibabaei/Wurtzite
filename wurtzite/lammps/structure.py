# +
from __future__ import annotations

import abc
import ctypes
import functools
from typing import Sequence

import numpy as np
from ase.calculators.lammps import convert

import wurtzite.lammps._backend as backend
from wurtzite.atomic_structure import AtomicStructure, DynamicStructure
from wurtzite.lammps._backend import lammps
from wurtzite.lammps.fix import Fix
from wurtzite.lammps.force_field import ForceField


class LAMMPS(DynamicStructure):
    _default_units = "real"
    _lmp: lammps
    _groups = 0

    @abc.abstractmethod
    def get_units(self) -> str:
        ...

    @abc.abstractmethod
    def get_types(self) -> dict[str, int]:
        ...

    def set_forcefield(self, ff: ForceField) -> None:
        self._lmp.commands_list(ff.get_commands(self.get_units(), self.get_types()))

    def set_fix(self, fix: Fix) -> None:
        self._lmp.commands_list(fix.get_commands(self.get_units()))

    def get_potential_energy(self, units: str = "ASE") -> float:
        self._lmp.command("run 0")
        e = self._lmp.get_thermo("pe")
        e = convert(e, "energy", self.get_units(), units)
        return e

    def get_type(self, symbol: str) -> int:
        return self.get_types()[symbol]

    def define_group(self, range: tuple[int, int], id: str | None = None) -> str:
        a, b = range
        if id is None:
            self._groups += 1
            id = f"g{self._groups}"
        cmd = f"group {id} id {a+1}:{b}"
        self._lmp.command(cmd)
        return id

    def gather(self, quantity, indices=None):

        if indices is None:
            _gather = self._lmp.gather_atoms
        else:
            _id = [i + 1 for i in indices]  # LAMMPS convention
            n = len(_id)
            subset = (ctypes.c_int * n)(*_id)
            _gather = functools.partial(
                self._lmp.gather_atoms_subset, ndata=n, ids=subset
            )

        if quantity in ("symbol", "symbols"):
            types = _gather("type", 0, 1)
            mapping = {t: s for s, t in self.get_types().items()}
            result = [mapping[t] for t in np.ctypeslib.as_array(types)]
        else:
            raise RuntimeError

        return result


class FullStyle(LAMMPS):
    def __init__(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        symbols: Sequence[str],
        pbc: bool | tuple[bool, bool, bool],
        units: str | None = None,
        log: str = "none",
        screen: str = "none",
    ):
        if units is None:
            units = self._default_units
        self._lmp, self._types = backend._create_lammps(
            positions, cell, symbols, pbc, units, log, screen
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
        backend._update_lammps_positions(
            self._lmp, self._pbc, self._cell, positions, self._units
        )
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
        struc: AtomicStructure, units: str | None = None, **kwargs
    ) -> FullStyle:
        return FullStyle(
            struc.get_positions(),
            struc.get_cell(),
            struc.get_chemical_symbols(),
            struc.get_pbc(),
            **kwargs,
        )
