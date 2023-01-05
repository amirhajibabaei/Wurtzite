# +
from __future__ import annotations

from typing import Any

import numpy as np
from ase.atoms import Atoms
from ase.calculators.lammps import convert
from ase.data import atomic_masses, chemical_symbols
from ase.geometry import wrap_positions
from lammps import lammps

from wurtzite.pairpot import write_table
from wurtzite.rotations import PrismRotation


class Lmp:
    def __init__(self, atoms: Atoms, units: str = "real"):

        lmp = lammps()

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
        prism = "0 {} 0 {} 0 {}  {} {} {}".format(*lower)
        lmp.command(f"region {region_id} prism  {prism} units box")

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
        for e, v in values.items():
            t = self._sym2type[e]
            self._lmp.command(f"set type {t} {quantity} {v}")

    def forcefield(self, pairpots, cutoff, charges):
        _table = "energy.table"
        keys = write_table(
            pairpots,
            self._units,
            _table,
            rmax=cutoff * 1.5,
            dr=0.01,
            cutoff=cutoff,
            shift=True,
        )
        N = []
        commands = []
        for pair, (id_, _N) in keys.items():
            t1 = self._sym2type[pair[0]]
            t2 = self._sym2type[pair[1]]
            N.append(_N)
            commands.append(f"pair_coeff {t1} {t2} table {_table} {id_} {cutoff}")
            commands.append(f"pair_coeff {t1} {t2} coul/long")
        (N,) = set(N)
        commands.insert(
            0, f"pair_style hybrid/overlay coul/long {cutoff} table linear {N}"
        )
        self._lmp.commands_list(commands)
        self._lmp.command("kspace_style pppm 0.00001")
        self._set("charge", charges)
