# +
import abc
from collections import Sequence

import numpy as np
from ase.data import atomic_numbers


class Layer:
    _symbols = "X"

    @property
    def num_sites(self):
        return len(self.xy_positions())

    @abc.abstractmethod
    def xy_positions(self):
        ...

    @abc.abstractmethod
    def xy_cell(self):
        ...

    @property
    def symbols(self):
        if type(self._symbols) == str:
            return self.num_sites * [self._symbols]
        else:
            return self._symbols

    @symbols.setter
    def symbols(self, _symbols):
        if isinstance(_symbols, str):
            assert _symbols in atomic_numbers, f"unknown symbol {_symbols}!"
        elif isinstance(_symbols, Sequence):
            assert (
                len(_symbols) == self.num_sites
            ), f"incorrect len of _symbols: {len(_symbols)}"
            unknown = [s for s in _symbols if s not in atomic_numbers]
            assert len(unknown) == 0, f"unknown symbols {unknown}!"
        else:
            raise RuntimeError("_symbols should be str or sequence")
        self._symbols = _symbols


class Shifted(Layer):
    def __init__(self, layer, tr, overwrite=None):
        self._layer = layer
        self._tr = np.asarray(tr)
        if overwrite is None:
            self._symbols = layer._symbols
        else:
            self.symbols = overwrite

    def xy_positions(self):
        return self._layer.xy_positions() + self._tr

    def xy_cell(self):
        return self._layer.xy_cell()


class Lattice(Layer):
    def __init__(self, uc, b, nxy, symbols="X"):
        uc = np.asarray(uc)
        b = np.asarray(b)
        nx, ny = nxy
        p = np.mgrid[0:nx, 0:ny].reshape(2, -1).T
        q = (p[:, None] + b).reshape(-1, 2)
        self._pos = (q[..., None] * uc).sum(axis=-2)
        self._cell = np.array([[nx], [ny]]) * uc
        self.symbols = symbols

    def xy_positions(self):
        return self._pos

    @abc.abstractmethod
    def xy_cell(self):
        return self._cell


class Cubic(Lattice):
    def __init__(self, a, nxy, symbols="X"):
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
        super().__init__(uc, b, nxy, symbols=symbols)


class Hexagonal(Lattice):
    def __init__(self, a, nxy, symbols="X"):
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
        super().__init__(uc, b, nxy, symbols=symbols)


class HexagonalCC(Lattice):
    def __init__(self, a, nxy, symbols="X"):
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
        super().__init__(uc, b, nxy, symbols=symbols)
