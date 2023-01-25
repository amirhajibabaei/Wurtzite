# +
from __future__ import annotations

try:
    # TODO:
    # import nglview
    import visard

    def view(struc):
        """
        Placeholder
        """

        if True:
            symbols = struc.get_chemical_symbols()
            opaque = [i for i, s in enumerate(symbols) if s == "X"]
            focus = [i for i, s in enumerate(symbols) if s != "X"]
            if len(opaque) == 0:
                focus = None  # type: ignore
        else:
            focus = None

        return visard.trajectory(struc.to_ase_atoms(), focus=focus, axes=True)

except ImportError:

    def view(struc):
        return None
