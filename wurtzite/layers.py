# +
import abc

import numpy as np


class Layer:
    @property
    def n(self):
        return len(self.xy_positions())

    @abc.abstractmethod
    def xy_positions(self):
        ...

    @abc.abstractmethod
    def xy_cell(self):
        ...


class Shifted(Layer):
    def __init__(self, layer, tr):
        self._layer = layer
        self._tr = np.asarray(tr)

    def xy_positions(self):
        return self._layer.xy_positions() + self._tr

    def xy_cell(self):
        return self._layer.xy_cell()


class Lattice(Layer):
    def __init__(self, uc, b, nx, ny):
        uc = np.asarray(uc)
        b = np.asarray(b)
        p = np.mgrid[0:nx, 0:ny].reshape(2, -1).T
        q = (p[:, None] + b).reshape(-1, 2)
        self._pos = (q[..., None] * uc).sum(axis=-2)
        self._cell = np.array([[nx], [ny]]) * uc

    def xy_positions(self):
        return self._pos

    @abc.abstractmethod
    def xy_cell(self):
        return self._cell


class Cubic(Lattice):
    def __init__(self, a, nx, ny):
        uc = (
            np.array(
                [
                    [1, 0],  #
                    [0, 1],  #
                ]
            )
            * a
        )

        b = np.array([[0, 0]])
        super().__init__(uc, b, nx, ny)


class Hexagonal(Lattice):
    def __init__(self, a, nx, ny):
        uc = (
            np.array(
                [
                    [1, 0],  #
                    [1 / 2, np.sqrt(3) / 2],  #
                ]
            )
            * a
        )
        b = np.array([[0, 0]])
        super().__init__(uc, b, nx, ny)


class HexagonalCC(Lattice):
    def __init__(self, a, nx, ny):
        uc = (
            np.array(
                [
                    [1, 0],  #
                    [0, np.sqrt(3)],  #
                ]
            )
            * a
        )
        b = np.array([[0, 0], [0.5, 0.5]])
        super().__init__(uc, b, nx, ny)
