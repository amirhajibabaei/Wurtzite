# +
from __future__ import annotations

try:
    # TODO:
    import nglview
    import visard
except ImportError:
    visard = None


class View(nglview.widget.NGLWidget):
    """
    Placeholder
    """

    pass


def view(struc) -> View | None:
    """
    Placeholder
    """
    if visard is None:
        return None
    else:
        return visard.trajectory(struc.to_ase_atoms(), axes=True)
