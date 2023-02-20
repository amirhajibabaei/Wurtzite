# +
from dataclasses import dataclass

from wurtzite.atomic_structure import PlaneStacking, WurtZite


@dataclass
class LatticeParam:
    """
    The crystal structure is wurtzite with
    alternating hexagonal Ag and I layers:

        Ag - I -- Ag - I ...
        | z1 | z2 | z1 | ...

    a: lattice constant for hexagonal layers
    z1: the smaller distance of Ag-I layers
    z2: the larger distance of Ag-I layers

    The default values are obtained from NPT
    MD averages at 300K.

    """

    a: float = 4.701159859488822
    z1: float = 0.8350589042773436
    z2: float = 2.930866754308036
    # z1_Ag = 0.28897763244509367
    # z1_I = 0.7073479272906041


def unitcell(param: LatticeParam) -> PlaneStacking:
    return WurtZite(param.a, param.z1, param.z2, ("Ag", "I"))
