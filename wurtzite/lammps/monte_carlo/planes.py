# +
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import lammps.constants as const
import numpy as np
from ase.calculators.lammps import convert

import wurtzite.tools as tools
from wurtzite.atomic_structure import PlaneStacking
from wurtzite.lammps.force_field import ForceField
from wurtzite.lammps.structure import FullStyle
from wurtzite.mpi import world


def plane_monte_carlo(
    stack: PlaneStacking,
    ff: ForceField,
    active_planes: Sequence[int],
    steps: int,
    temp: float,
    every: int = 1,
    attempts: int | None = None,
    random_state: int | np.random.RandomState = 20230126,
    log: str | None = None,
) -> tuple[PlaneStacking, PlaneStacking, float, dict[str, float]]:
    """
    Lattice Monte Carlo for a "PlaneStacking" with swaps limited
    to pair of atoms are which belong to the same "AtomicPlane".
    The planes which are not listed in "active_planes" are frozen.

    if attempts = None -> attempts = min(count[a], count[b]) (= 1 sweep)

    Returns: final, mimimum, minimum energy, swap ratio

    """

    struc = FullStyle.from_atomic_structure(stack)
    # to ensure "table pair styles" are ready:
    struc._lmp.comm.Barrier()  # TODO: no _lmp!
    struc.set_forcefield(ff)

    struc._lmp.commands_list(  # TODO: no _lmp!
        ["thermo_style custom step pe", "thermo 1", "run 0"]
    )

    rng = tools.get_random_state(random_state)

    swap_fixes = []
    for plane in active_planes:
        index_range = stack.index_range(plane)
        group = struc.define_group(index_range)
        count = stack[plane].count()
        symbols = count.keys()
        pairs = tools.pairings(symbols, self_interaction=False)
        for a, b in pairs:
            X = min(count[a], count[b]) if attempts is None else attempts
            t1 = struc.get_type(a)
            t2 = struc.get_type(b)
            fid = f"swap_{plane}_{a}_{b}"
            seed = rng.randint(2**16 - 1)
            cmd = (
                f"fix {fid} {group} atom/swap "
                f"{every} {X} {seed} {temp} "
                f"types {t1} {t2} ke no semi-grand no "
            )
            struc._lmp.command(cmd)  # TODO: no _lmp!
            swap_fixes.append(fid)

    def gather_symbols() -> tuple[tuple[str, ...]]:
        symbols = tuple(
            struc.gather("symbol", indices=range(*stack.index_range(i)))
            for i in active_planes
        )
        return symbols  # type: ignore

    def callback(optim, step, nlocal, tag, pos, fext):
        e = struc._lmp.get_thermo("pe")  # TODO: no _lmp!
        if e < optim.energy:
            optim.energy = e
            optim.symbols = gather_symbols()
            if log is not None and world.Get_rank() == 0:
                _step = struc._lmp.extract_global("ntimestep")
                with open(log, "a") as of:
                    of.write(f"\n{_step} energy = {e} \n{optim.symbols}\n")

    @dataclass
    class State:
        energy: float
        symbols: tuple[tuple[str, ...]]

    # TODO: no _lmp!
    optim = State(struc._lmp.get_thermo("pe"), gather_symbols())
    struc._lmp.command("fix ext1 all external pf/callback 1 1")
    struc._lmp.set_fix_external_callback("ext1", callback, caller=optim)
    struc._lmp.command(f"run {steps}")

    # outputs
    final = stack.with_planes_symbols(active_planes, gather_symbols())
    minimum = stack.with_planes_symbols(active_planes, optim.symbols)
    mimimum_energy = convert(optim.energy, "energy", struc.get_units(), "ASE")
    swap_ratio = _get_swap_ratio(struc._lmp, swap_fixes)
    return final, minimum, mimimum_energy, swap_ratio


def _get_swap_ratio(lmp, swap_fixes: Sequence[str]) -> dict[str, float]:
    ratio: dict[str, float] = {}
    for fid in swap_fixes:
        trials = lmp.numpy.extract_fix(
            fid, const.LMP_STYLE_GLOBAL, const.LMP_TYPE_VECTOR, nrow=0
        )
        successes = lmp.numpy.extract_fix(
            fid, const.LMP_STYLE_GLOBAL, const.LMP_TYPE_VECTOR, nrow=1
        )
        ratio[fid] = successes / trials
    return ratio
