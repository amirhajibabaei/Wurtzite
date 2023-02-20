# +
from wurtzite.atomic_structure import AtomicStructure
from wurtzite.lammps.force_field import CoulTableHybrid, ForceField, pairings
from wurtzite.lammps.structure import FullStyle
from wurtzite.pair_potential import ZeroPot, rePRV


def get_force_field(
    q: float = 0.5815, cutoff: float = 10.0, pppm: float = 1e-3, dr: float = 0.01
) -> ForceField:
    charges = {"Ag": q, "I": -q, "X": 0.0}
    pairpots = {
        pair: ZeroPot() if "X" in pair else rePRV(pair) for pair in pairings(charges)
    }

    ff = CoulTableHybrid(
        pairpots, cutoff, charges, kspace_style=f"pppm {pppm}", table_dr=dr
    )
    return ff


forcefield = get_force_field()


def get_poential_energy(struc: AtomicStructure, units: str = "ASE") -> float:
    x = FullStyle.from_atomic_structure(struc)
    x.set_forcefield(forcefield)
    return x.get_potential_energy(units=units)
