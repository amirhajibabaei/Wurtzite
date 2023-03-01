# +
from __future__ import annotations

import os

from wurtzite.atomic_structure import AtomicStructure
from wurtzite.lammps.force_field import CoulTableHybrid, ForceField, pairings
from wurtzite.lammps.structure import FullStyle
from wurtzite.mpi import world
from wurtzite.pair_potential import ZeroPot, rePRV


def get_force_field(
    q: float = 0.5815,
    cutoff: float = 10.0,
    pppm: float = 1e-3,
    dr: float = 0.01,
    pair_write: str | None = None,
) -> ForceField:
    charges = {"Ag": q, "I": -q, "X": 0.0}
    pairpots = {
        pair: ZeroPot() if "X" in pair else rePRV(pair) for pair in pairings(charges)
    }

    if pair_write is not None and world.Get_rank() == 0:
        os.system("rm -f _table.txt")
    world.Barrier()
    ff = CoulTableHybrid(
        pairpots,
        cutoff,
        charges,
        kspace_style=f"pppm {pppm}",
        table_dr=dr,
        pair_write=pair_write,
    )
    return ff


forcefield = get_force_field()


def get_poential_energy(
    struc: AtomicStructure, units: str = "ASE", ff: ForceField | None = None
) -> float:
    x = FullStyle.from_atomic_structure(struc)
    if ff is None:
        ff = forcefield
    x.set_forcefield(ff)
    return x.get_potential_energy(units=units)
