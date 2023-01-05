# +
from wurtzite.stack import WurtZite, WurtZiteCC


def AgI_wurtzite0001(nxyz, vacuum=20, cubic_cell=False, adjust_surfaces=False):
    """
    The parameters are obtained from
    classical MD simulations at 300 K.
    """
    cell_a = 4.701159859488822
    # cell_c = 7.531811086118183 # = 2*(z1+z2)
    cell_z1 = 0.8350589042773436
    cell_z2 = 2.930866754308036
    z1_surf_Ag = 0.28897763244509367
    z1_surf_I = 0.7073479272906041

    _WurtZite = WurtZiteCC if cubic_cell else WurtZite

    if adjust_surfaces:
        surf_z1 = (z1_surf_Ag, z1_surf_I)
    else:
        surf_z1 = None

    stack = _WurtZite(
        cell_a, cell_z1, cell_z2, nxyz, ["Ag", "I"], surf_z1=surf_z1, vacuum=vacuum
    )

    return stack
