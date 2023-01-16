# +
import ctypes

import lammps.constants as const
import numpy as np
from ase.calculators.lammps import convert

from wurtzite.atomic_structure import AtomicStructure
from wurtzite.lammps.force_field import ForceField
from wurtzite.lammps.structure import FullStyle


def lattice_monte_carlo(
    structure: AtomicStructure,
    forcefield: ForceField,
    index_range: tuple[int, int],
    pair: tuple[str, str],
    temperature: float,
    steps: int,
    random_seed: int = 57465456,
    every: int = 1,
    attempts: int = 1,
) -> tuple[float, float, tuple[str, ...]]:

    # create a LAMMPS instance
    struc = FullStyle.from_atomic_structure(structure)
    struc.set_forcefield(forcefield)
    lmp = struc._lmp  # TODO: it is not recommended to use hidden attributes
    sym2type = struc.get_types()
    type2sym = {t: s for s, t in sym2type.items()}
    symbols = struc.get_chemical_symbols()
    subset_symbols = [symbols[i] for i in range(*index_range)]

    # Initial state:
    lmp.command("run 0")
    optimum = [
        lmp.get_thermo("pe"),
        np.array([sym2type[s] for s in subset_symbols]),
    ]

    # Mote-Carlo setups
    fix_swap = "f_swap"
    group = "active"
    lmp_pair_types = " ".join([str(sym2type[s]) for s in pair])
    lmp.commands_list(
        [
            "thermo_style custom pe ",
            "thermo 1",
            f"group {group} id {index_range[0]+1}:{index_range[1]}",
            f"fix {fix_swap} {group} atom/swap "
            f"{every} {attempts} {random_seed} {temperature} "
            f"types {lmp_pair_types} ke no semi-grand no ",
        ]
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
    ntry = lmp.numpy.extract_fix(
        fix_swap, const.LMP_STYLE_GLOBAL, const.LMP_TYPE_VECTOR, nrow=0
    )
    nsuccess = lmp.numpy.extract_fix(
        fix_swap, const.LMP_STYLE_GLOBAL, const.LMP_TYPE_VECTOR, nrow=1
    )

    ratio = nsuccess / ntry
    energy_opt = convert(optimum[0], "energy", lmp.extract_global("units"), "ASE")
    symbols_opt = tuple((type2sym[t] for t in optimum[1]))
    return ratio, energy_opt, symbols_opt
