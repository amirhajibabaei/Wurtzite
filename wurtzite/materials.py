# +
from __future__ import annotations

from wurtzite.atomic_structure import PlaneStacking, WurtZite, WurtZite2


def AgI_wurtzite0001(
    repeat: int | tuple[int, int, int],
    *,
    cubic_cell: bool = False,
    bulk_termination: bool = True,
    vacuum: float | None = None,
) -> PlaneStacking:
    """
    The parameters are obtained from
    classical MD simulations at 300 K.
    """

    a = 4.701159859488822
    # c = 7.531811086118183 # = 2*(z1+z2)
    z1 = 0.8350589042773436
    z2 = 2.930866754308036
    z1_Ag = 0.28897763244509367
    z1_I = 0.7073479272906041

    _WurtZite = WurtZite2 if cubic_cell else WurtZite
    stack = _WurtZite(a, z1, z2, ("Ag", "I")).repeat(repeat)

    if not bulk_termination:
        stack = stack.with_spacing(0, z1_Ag)
        stack = stack.with_spacing(-2, z1_I)

    if vacuum is not None:
        stack = stack.with_spacing(-1, vacuum)

    return stack


def AgI_wurtzite0001_re_2x2(nz: int, *, vacancy: str = "X", **kwargs) -> PlaneStacking:
    """
    Returns a (2, 2, nz) slab with 1/4 of atoms replaced
    with vacancy on the two Ag-/I-terminated surfaces.
    """
    ideal = AgI_wurtzite0001((2, 2, nz), **kwargs)
    re_1st = ideal[0].with_chemical_symbols(f"Ag Ag Ag {vacancy}".split())
    re_last = ideal[-1].with_chemical_symbols(f"I I I {vacancy}".split())
    re = ideal.with_plane(0, re_1st).with_plane(-1, re_last)
    return re
