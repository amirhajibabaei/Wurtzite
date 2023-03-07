# +
from __future__ import annotations

from wurtzite.atomic_structure import AtomicStructure, PlaneStacking
from wurtzite.projects.AgI.bulk import LatticeParam, unitcell
from wurtzite.projects.AgI.force_field import get_force_field, get_poential_energy
from wurtzite.projects.AgI.reconstruction import Reconstruction
from wurtzite.projects.AgI.special import vacancy

# using accurate long-range forces
_forcefield = get_force_field(pppm=1e-6)
_param = LatticeParam()


def per_atom_energy(struc: AtomicStructure | Reconstruction) -> float:
    if isinstance(struc, Reconstruction):
        struc = struc.get_stacking(_param)
    e = get_poential_energy(struc, ff=_forcefield) / len(struc)
    return e


_bulk = unitcell(_param)
_bulk_energy = per_atom_energy(_bulk)
_re_vacancy_energy = per_atom_energy(vacancy)


def re_energy(struc: AtomicStructure | Reconstruction) -> float:
    """
    Reconstruction with array of vacancies is
    considered as reference.
    """
    e = per_atom_energy(struc) - _re_vacancy_energy
    return e


def surface_energy(struc: PlaneStacking | Reconstruction) -> float:
    """
    Surface energy
    """
    if isinstance(struc, Reconstruction):
        struc = struc.get_stacking(_param)
    e = get_poential_energy(struc, ff=_forcefield)
    b = _bulk_energy
    N = len(struc)
    gamma = (e - N * b) / (2 * struc.get_surface_area())
    return gamma
