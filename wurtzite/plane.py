# +
import abc
from collections import Counter
from collections.abc import Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers


class Plane:
    @abc.abstractmethod
    def xy_positions(self):
        ...

    @abc.abstractmethod
    def xy_cell(self):
        ...

    def __len__(self):
        return len(self.xy_positions())

    def get_positions(self, z):
        xy = self.xy_positions()
        z = np.full(xy.shape[0], z)
        return np.c_[xy, z]

    def translate(self, tr):
        return ShiftedPlane(self, tr)


class CustomPlane(Plane):
    def __init__(self, xy_positions, xy_cell):
        self._pos = xy_positions
        self._cell = xy_cell

    def xy_positions(self):
        return self._pos

    def xy_cell(self):
        return self._cell


class LatticePlane(Plane):
    def __init__(self, vectors, basis, nxy):
        self._vectors = np.asarray(vectors)
        self._basis = np.asarray(basis)
        self._nxy = nxy
        self._pos = None
        self._cell = None

    def xy_positions(self):
        if self._pos is None:
            nx, ny = self._nxy
            p = np.mgrid[0:nx, 0:ny].reshape(2, -1).T
            q = (p[:, None] + self._basis).reshape(-1, 2)
            self._pos = (q[..., None] * self._vectors).sum(axis=-2)
        return self._pos

    def xy_cell(self):
        if self._cell is None:
            nxy = np.asarray(self._nxy).reshape(2, 1)
            self._cell = nxy * self._vectors
        return self._cell


class CubicPlane(LatticePlane):
    def __init__(self, a, nxy):
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
        super().__init__(uc, b, nxy)


class HexagonalPlane(LatticePlane):
    def __init__(self, a, nxy):
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
        super().__init__(uc, b, nxy)


class HexagonalPlaneCC(LatticePlane):
    def __init__(self, a, nxy):
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
        super().__init__(uc, b, nxy)


class ShiftedPlane(Plane):
    def __init__(self, plane, tr):
        self._plane = plane
        self._tr = np.asarray(tr)
        self._pos = None

    def xy_positions(self):
        if self._pos is None:
            self._pos = self._plane.xy_positions() + self._tr
        return self._pos

    def xy_cell(self):
        return self._plane.xy_cell()


class AtomicPlane:
    def __init__(self, plane, atoms="X"):

        if type(atoms) == str:
            assert atoms in atomic_numbers
        elif isinstance(atoms, Sequence):
            assert len(atoms) == len(plane)
            unknown = [a for a in set(atoms) if a not in atomic_numbers]
            assert len(unknown) == 0
        else:
            raise RuntimeError("Invalid atoms!")

        self._plane = plane
        self._atoms = atoms

    @property
    def atoms(self):
        if type(self._atoms) == str:
            return len(self._plane) * [self._atoms]
        else:
            return self._atoms

    @property
    def xy_cell(self):
        return self._plane.xy_cell()

    def get_positions(self, z):
        return self._plane.get_positions(z)

    def replace_atoms(self, atoms):
        return AtomicPlane(self._plane, atoms)

    def __repr__(self):
        return f"AtomicPlane: {Counter(self.atoms)}"

    def as_ase_atoms(self, vacuum=10.0):
        cell_xy = np.c_[self.xy_cell, [0, 0]]
        cell = np.r_[cell_xy, [[0, 0, 2 * vacuum]]]
        atoms = Atoms(
            symbols=self.atoms,
            positions=self.get_positions(vacuum),
            cell=cell,
            pbc=True,
        )
        return atoms
