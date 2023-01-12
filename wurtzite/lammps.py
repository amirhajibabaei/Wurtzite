# +
from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
from ase.calculators.lammps import convert
from ase.data import atomic_masses, chemical_symbols
from ase.geometry import wrap_positions
from lammps import lammps

from wurtzite.atomic_structure import AtomicStructure
from wurtzite.pairpot import PairPot, write_lammps_table
from wurtzite.rotations import PrismRotation


class LAMMPS:
    def __init__(self, structure: AtomicStructure, units: str = "real"):

        lmp = lammps(cmdargs="-screen none".split())

        # units, style, boundary:
        pbc = structure.get_pbc()
        lmp.commands_list([f"units {units}", "atom_style full", _pbc_to_str(pbc)])

        # rotate coordinates to LAMMPS "prism" style
        prism, positions = ase_cell_to_lammps_prism(
            pbc, structure.get_cell(), structure.get_positions(), units
        )

        # define region
        region_id = "cell"
        prism = "0 {} 0 {} 0 {} {} {} {}".format(*prism)
        lmp.command(f"region {region_id} prism {prism} units box")

        # create box
        numbers = structure.get_atomic_numbers()
        unique = np.unique(numbers)  # auto-sorted
        lmp.command(f"create_box {len(unique)} {region_id}")

        # create structure
        num2type = {z: i + 1 for i, z in enumerate(unique)}
        types = list(map(num2type.get, numbers))
        lmp.create_atoms(
            n=len(numbers),
            id=None,
            type=types,
            x=positions.reshape(-1),
            v=None,
            image=None,
            shrinkexceed=False,
        )

        # masses
        sym2type = dict()
        for z, t in num2type.items():
            m = convert(atomic_masses[z], "mass", "ASE", units)
            lmp.command(f"mass {t} {m}")
            sym2type[chemical_symbols[z]] = t

        #
        self._units = units
        self._lmp = lmp
        self._num2type = num2type
        self._sym2type = sym2type

    # In-place:

    def _set(self, quantity: str, values: dict[str, Any]) -> None:
        for e, _v in values.items():
            t = self._sym2type[e]
            v = convert(_v, quantity, "ASE", self._units)
            self._lmp.command(f"set type {t} {quantity} {v}")

    def set_charges_(self, charges: dict[str, float]) -> None:
        self._set("charge", charges)

    def set_pair_coeffs_(
        self,
        pairpots: dict[tuple[str, str], PairPot],
        cutoff: float,
        kspace_style: str = "pppm 1e-4",
        dr: float = 0.01,
        verbose: bool = False,
    ):
        # write pairpots to table
        _table = "_pairs.table"
        N, keys = write_lammps_table(
            pairpots,
            self._units,
            _table,
            rmax=cutoff * 1.5,
            dr=dr,
            cutoff=cutoff,
            shift=True,
        )

        #
        cutoff = convert(cutoff, "distance", "ASE", self._units)
        commands = [f"pair_style hybrid/overlay coul/long {cutoff} table linear {N}"]
        for pair, key in keys.items():
            t1 = self._sym2type[pair[0]]
            t2 = self._sym2type[pair[1]]
            commands.append(f"pair_coeff {t1} {t2} table {_table} {key} {cutoff}")
            commands.append(f"pair_coeff {t1} {t2} coul/long")

        if kspace_style is not None:
            commands.append(f"kspace_style {kspace_style}")

        if verbose:
            print("\n".join(commands))

        self._lmp.commands_list(commands)

    def update_positions_(
        self, pbc: tuple[bool, bool, bool], cell: np.ndarray, positions: np.ndarray
    ) -> None:

        prism, positions = ase_cell_to_lammps_prism(pbc, cell, positions, self._units)

        # update box
        box = " ".join([f"{a} final 0 {b}" for a, b in zip(["x", "y", "z"], prism[:3])])
        tilt = " ".join(
            [f"{a} final {b}" for a, b in zip(["xy", "xz", "yz"], prism[3:])]
        )
        self._lmp.command(f"change_box all {box} {tilt} {_pbc_to_str(pbc)}")

        # update positions
        self._lmp.scatter_atoms("x", 1, 3, array_c_ptr(positions.reshape(-1)))

    # Thermo:

    def get_potential_energy(self, units: str = "ASE") -> float:
        self._lmp.command("run 0")
        e = self._lmp.get_thermo("pe")
        e = convert(e, "energy", self._units, units)
        return e


def ase_cell_to_lammps_prism(
    pbc: tuple[bool, bool, bool], cell: np.ndarray, positions: np.ndarray, units: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    returns converted cell, positions.
    """
    # rotate coordinates to LAMMPS "prism" style
    rotation = PrismRotation(cell)
    rot_cell = rotation(cell)
    rot_positions = rotation(positions)

    # wrap
    # TODO: don't wrap! use image numbers.
    rot_positions = wrap_positions(rot_positions, rot_cell, pbc)

    # convert units
    con_cell = convert(rot_cell, "distance", "ASE", units)
    con_positions = convert(rot_positions, "distance", "ASE", units)

    # prism
    assert np.allclose(con_cell.flat[[1, 2, 5]], 0)
    prism = con_cell.flat[[0, 4, 8, 3, 6, 7]]
    return prism, con_positions


def array_c_ptr(arr):
    ptr = ctypes.POINTER(ctypes.c_double)
    return arr.astype(np.float64).ctypes.data_as(ptr)


def _pbc_to_str(pbc: tuple[bool, bool, bool]) -> str:
    return "boundary " + " ".join(["p" if b else "f" for b in pbc])
