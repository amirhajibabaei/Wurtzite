# +
from __future__ import annotations

from typing import Any

import numpy as np
from ase.atoms import Atoms
from ase.calculators.lammps import convert
from ase.data import atomic_masses, chemical_symbols
from ase.geometry import wrap_positions
from lammps import lammps

from wurtzite.pairpot import write_lammps_table
from wurtzite.rotations import PrismRotation


class LAMMPS:
    def __init__(self, atoms: Atoms, units: str = "real"):

        lmp = lammps(cmdargs="-screen none".split())

        # units, style, boundary:
        boundary = " ".join(["p" if b else "f" for b in atoms.pbc])
        lmp.commands_list([f"units {units}", "atom_style full", f"boundary {boundary}"])

        # rotate coordinates to LAMMPS "prism" style
        rotation = PrismRotation(atoms.cell)
        cell = rotation(atoms.cell)
        cell = convert(cell, "distance", "ASE", units)
        positions = rotation(atoms.positions)
        positions = wrap_positions(positions, cell, atoms.pbc)
        positions = convert(positions, "distance", "ASE", units)

        # define region
        region_id = "cell"
        lower = cell.flat[[0, 4, 8, 3, 6, 7]]
        prism = "0 {} 0 {} 0 {} {} {} {}".format(*lower)
        lmp.command(f"region {region_id} prism {prism} units box")

        # create box
        unique = np.unique(atoms.get_atomic_numbers())  # auto-sorted
        lmp.command(f"create_box {len(unique)} {region_id}")

        # create atoms
        num2type = {z: i + 1 for i, z in enumerate(unique)}
        types = list(map(num2type.get, atoms.numbers))
        lmp.create_atoms(
            n=len(atoms),
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
        self._atoms = atoms
        self._lmp = lmp
        self._cell = cell
        self._positions = positions
        self._num2type = num2type
        self._sym2type = sym2type

    def _set(self, quantity: str, values: dict[str, Any]) -> None:
        for e, _v in values.items():
            t = self._sym2type[e]
            v = convert(_v, quantity, "ASE", self._units)
            self._lmp.command(f"set type {t} {quantity} {v}")

    def set_charges(self, charges):
        self._set("charge", charges)

    def set_pair_coeffs(
        self, pairpots, cutoff, kspace_style="pppm 1e-4", verbose=False
    ):
        # write pairpots to table
        _table = "_pairs.table"
        N, keys = write_lammps_table(
            pairpots,
            self._units,
            _table,
            rmax=cutoff * 1.5,
            dr=0.01,
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

    def get_potential_energy(self):
        self._lmp.command("run 0")
        e = self._lmp.get_thermo("pe")
        e = convert(e, "energy", self._units, "ASE")
        return e
