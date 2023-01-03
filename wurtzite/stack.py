# +
import abc

import numpy as np
from ase import Atoms

from .layers import Hexagonal, HexagonalCC, Shifted


class Stack:
    def __init__(self, symbols, layers, spacings, vacuum=20.0):
        assert len(symbols) == len(layers) == len(spacings) + 1
        self._symbols = symbols
        self.layers = layers
        self.spacings = spacings
        self.vacuum = vacuum

    @property
    def cell(self):
        cells = [layer.xy_cell() for layer in self.layers]
        assert np.allclose(np.stack(cells) - cells[0], 0)
        cell_xy = np.c_[cells[0], [0, 0]]
        cell_z = sum(self.spacings) + 2 * self.vacuum
        cell = np.r_[cell_xy, [[0, 0, cell_z]]]
        return cell

    @property
    def symbols(self):
        symbols = []
        for layer, symbol in zip(self.layers, self._symbols):
            symbols.extend(layer.n * [symbol])
        return symbols

    @property
    def positions(self):
        xy, z = [], []
        spacings = [*self.spacings, 0]
        _z = self.vacuum
        for layer, delta in zip(self.layers, spacings):
            xy.append(layer.xy_positions())
            z.append(layer.n * [_z])
            _z += delta
        xyz = np.c_[np.concatenate(xy), np.concatenate(z)]
        return xyz

    def as_atoms(self):
        atoms = Atoms(
            symbols=self.symbols, positions=self.positions, cell=self.cell, pbc=True
        )
        return atoms


class _WurtZite(Stack):
    def __init__(
        self, cell_a, cell_z1, cell_z2, nxyz, symbols, vacuum=20.0, surf_z1=None
    ):
        nx, ny, nz = nxyz
        A, B = symbols
        a, b = self._hexagonal_layers(cell_a, nx, ny)
        layers = nz * [a, b, b, a]
        symbols = nz * [A, B, A, B]
        spacings = (nz * [cell_z1, cell_z2, cell_z1, cell_z2])[:-1]
        if surf_z1 is not None:
            spacings[0], spacings[-1] = surf_z1
        super().__init__(symbols, layers, spacings, vacuum=vacuum)

    @abc.abstractmethod
    def _hexagonal_layers(self, cell_a, nx, ny):
        ...


class WurtZiteCC(_WurtZite):
    def _hexagonal_layers(self, cell_a, nx, ny):
        a = HexagonalCC(cell_a, nx, ny)
        b = Shifted(a, [0, cell_a / np.sqrt(3)])
        return a, b


class WurtZite(_WurtZite):
    def _hexagonal_layers(self, cell_a, nx, ny):
        a = Hexagonal(cell_a, nx, ny)
        b = Shifted(a, [cell_a, cell_a / np.sqrt(3)])
        return a, b
