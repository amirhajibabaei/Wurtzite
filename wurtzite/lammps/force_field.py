# +
from __future__ import annotations

import abc
from typing import Sequence

from ase.calculators.lammps import convert

from wurtzite.lammps.table_io import write_lammps_table
from wurtzite.pair_potential import PairPotential
from wurtzite.tools import pairings


class ForceField(abc.ABC):

    # For children:

    @abc.abstractmethod
    def get_pair_style(self, units: str) -> str:
        ...

    @abc.abstractmethod
    def get_pair_coeff(self, units: str, types: dict[str, int]) -> Sequence[str]:
        """
        types:
            chemical symbol to lammps-type mapping e.g. {"H": 1, "O": 2, ...}
        """
        ...

    # Derived:

    def get_commands(self, units: str, types: dict[str, int]) -> Sequence[str]:
        return [self.get_pair_style(units), *self.get_pair_coeff(units, types)]

    # Utility:

    def _set(
        self,
        units: str,
        types: dict[str, int],
        quantity: str,
        values: dict[str, float],
    ) -> Sequence[str]:
        """
        types:
            chemical symbol to lammps-type mapping e.g. {"H": 1, "O": 2, ...}
        values:
            chemical symbol to values mapping e.g. charge {"H": +1, "O": -2, ...}
        """
        commands = []
        for symbol, type_ in types.items():
            v = convert(values[symbol], quantity, "ASE", units)
            commands.append(f"set type {type_} {quantity} {v}")
        return commands


class CoulTableHybrid(ForceField):
    def __init__(
        self,
        pairpots: dict[tuple[str, str], PairPotential],
        cutoff: float,
        charges: dict[str, float],
        kspace_style: str | None = "pppm 1e-4",
        table_dr: float = 0.01,
        table_shift: bool = True,
        table_units: str = "real",
        pair_write: str | None = None,
    ):

        # write pairpots to table
        name, N, keys = write_lammps_table(
            pairpots,
            table_units,
            rmax=cutoff * 1.5,
            dr=table_dr,
            cutoff=cutoff,
            shift=table_shift,
        )

        self._table_name = name
        self._table_N = N
        self._table_keys = keys
        self._cutoff = cutoff
        self._charges = charges
        self._kspace = kspace_style
        self._pair_write = pair_write

    def get_pair_style(self, units: str) -> str:
        cutoff = convert(self._cutoff, "distance", "ASE", units)
        return (
            f"pair_style hybrid/overlay coul/long {cutoff} table linear {self._table_N}"
        )

    def get_pair_coeff(self, units: str, types: dict[str, int]) -> Sequence[str]:
        cutoff = convert(self._cutoff, "distance", "ASE", units)
        commands = []
        _pair_write = []
        for a, b in pairings(types):
            t1 = types[a]
            t2 = types[b]
            try:
                key = self._table_keys[(a, b)]
            except KeyError:
                key = self._table_keys[(b, a)]
            except KeyError:
                raise

            commands.append(
                f"pair_coeff {t1} {t2} table {self._table_name} {key} {cutoff}"
            )
            commands.append(f"pair_coeff {t1} {t2} coul/long")
            _pair_write.append(
                f"pair_write {t1} {t2} {self._table_N} "
                f"r {0.1} {cutoff} _table.txt {key} "
                # f"{self._charges[a]} {self._charges[b]}"
                f"0 0"
            )

        commands.extend(self._set(units, types, "charge", self._charges))

        if self._kspace is not None:
            commands.append(f"kspace_style {self._kspace}")
        if self._pair_write is not None:
            commands.extend(_pair_write)
        return commands
