# +
from __future__ import annotations

import ctypes

import lammps.constants as const
import numpy as np
from ase.calculators.lammps import convert

from wurtzite.atomic_structure import AtomicStructure
from wurtzite.lammps.force_field import ForceField
from wurtzite.lammps.structure import FullStyle
from wurtzite.tools import pairings


def lattice_monte_carlo(
    structure: AtomicStructure,
    forcefield: ForceField,
    temperature: float,
    steps: int,
    *,
    pairs: tuple[tuple[str, str], ...] | None = None,
    index_range: tuple[int, int] | None = None,
    random_seed: int = 57465456,  # TODO: use random state
    every: int = 1,
    attempts: int = 1,
) -> tuple[dict[tuple[str, str], float], float, tuple[str, ...]]:
    """
    Returns:
        acceptance_ratio
        minimum_energy (eV)
        optimal_arrangement (symbols)

    """

    # create a LAMMPS instance
    struc = FullStyle.from_atomic_structure(structure)
    struc.set_forcefield(forcefield)
    lmp = struc._lmp  # TODO: it is not recommended to use the hidden attributes!
    sym2type = struc.get_types()
    type2sym = {t: s for s, t in sym2type.items()}
    symbols = struc.get_chemical_symbols()
    if index_range is None:
        index_range = (0, len(struc))
    subset_symbols = [symbols[i] for i in range(*index_range)]

    # Initial state:
    lmp.commands_list(["thermo_style custom step pe ", "thermo 1", "run 0"])
    optimum = [
        lmp.get_thermo("pe"),
        np.array([sym2type[s] for s in subset_symbols]),
    ]

    # Mote-Carlo setups
    fix_swap = "f_swap"
    group = "active"
    lmp.command(f"group {group} id {index_range[0]+1}:{index_range[1]}")
    if pairs is None:
        pairs = pairings(set(subset_symbols), self_interaction=False)
    for (a, b) in pairs:
        assert a != b
        lmp_pair_types = f"{sym2type[a]} {sym2type[b]}"
        lmp.command(
            f"fix {fix_swap}_{a}_{b} {group} atom/swap "
            f"{every} {attempts} {random_seed} {temperature} "
            f"types {lmp_pair_types} ke no semi-grand no "
        )

    # Extenral setups
    ids = list(range(index_range[0] + 1, index_range[1] + 1))
    nsub = len(ids)
    subset = (ctypes.c_int * nsub)(*ids)

    def callback(optimum, step, nlocal, tag, pos, fext):
        e = lmp.get_thermo("pe")
        if optimum[0] is None or e < optimum[0]:
            optimum[0] = e
            optimum[1] = np.ctypeslib.as_array(
                lmp.gather_atoms_subset("type", 0, 1, nsub, subset)
            )
            # charge: ("q", 1, 1, nsub, subset)
            # positions ("x", 1, 3, nsub, subset)
        # print(step, e, optimum[0])

    fix_ext = "f_ext"
    lmp.command(f"fix {fix_ext} all external pf/callback 1 1")
    lmp.set_fix_external_callback(fix_ext, callback, caller=optimum)

    # Run
    lmp.command(f"run {steps}")

    # Return:
    ratio: dict[tuple[str, str], float] = {}
    for (a, b) in pairs:
        fid = f"{fix_swap}_{a}_{b}"
        ntry = lmp.numpy.extract_fix(
            fid, const.LMP_STYLE_GLOBAL, const.LMP_TYPE_VECTOR, nrow=0
        )
        nsuccess = lmp.numpy.extract_fix(
            fid, const.LMP_STYLE_GLOBAL, const.LMP_TYPE_VECTOR, nrow=1
        )
        ratio[(a, b)] = nsuccess / ntry
    energy_opt = convert(optimum[0], "energy", lmp.extract_global("units"), "ASE")
    symbols_opt = tuple((type2sym[t] for t in optimum[1]))
    return ratio, energy_opt, symbols_opt
