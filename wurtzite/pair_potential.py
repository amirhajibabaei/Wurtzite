# +
from __future__ import annotations

import abc

import numpy as np


class PairPotential(abc.ABC):
    """
    ASE units are assumed
        distance: A (Angstrom)
        energy: eV
        force: eV/A
    """

    @abc.abstractmethod
    def energy_and_force(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        input: r (A)
        returns: e (eV), f (ev/A)
        """
        ...


class ZeroPot(abc.ABC):
    def energy_and_force(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros_like(r), np.zeros_like(r)


class _rePRV(PairPotential):
    """
    Reparameterized Parrinello–Rahman–Vashista potential
    """

    def __init__(self, eta, H, P, C):
        """
              |  eta      H      P       C
        ----------------------------------
        Ag-Ag |  11     0.16      0      0
        Ag-I  |   9   1310.0   14.9      0
        I-I   |   7   5328.0   29.8   84.5

        """
        self._params = eta, H, P, C

    def energy_and_force(self, r):
        eta, H, P, C = self._params
        e = H / r**eta - C / r**6 - P / r**4
        f = eta * H / r ** (eta + 1) - 6 * C / r**7 - 4 * P / r**5
        return e, f


class rePRV(_rePRV):

    parameters = {
        ("Ag", "Ag"): (11, 0.16, 0, 0),
        ("Ag", "I"): (9, 1310.0, 14.9, 0),
        ("I", "I"): (7, 5328.0, 29.8, 84.5),
    }

    def __init__(self, pair):
        a, b = pair
        try:
            par = self.parameters[(a, b)]
        except KeyError:
            par = self.parameters[(b, a)]
        except KeyError:
            raise RuntimeError(f"rePRV is not defined for {a}, {b}")
        super().__init__(*par)
