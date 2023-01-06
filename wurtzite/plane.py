# +
import abc
from collections import Counter
from collections.abc import Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers


class Plane(abc.ABC):
    @abc.abstractmethod
    def xy_positions(self) -> "np.ndarray":  # shape: [:, 2]
        ...

    @abc.abstractmethod
    def xy_cell(self) -> "np.ndarray":  # shape: [2, 2]
        ...

    def __len__(self):
        return len(self.xy_positions())

    def xyz_positions(self, z: float) -> "np.ndarray":  # shape: [:, 3]
        xy = self.xy_positions()
        z = np.full(xy.shape[0], z)
        return np.c_[xy, z]

    def translate(self, tr) -> "Plane":
        return ShiftedPlane(self, tr)

    def repeat(self, rxy):
        return RepeatedPlane(self, rxy)


class CustomPlane(Plane):
    def __init__(self, xy_positions, xy_cell):
        self._pos = xy_positions
        self._cell = xy_cell

    def xy_positions(self):
        return self._pos

    def xy_cell(self):
        return self._cell


class LatticePlane(Plane):
    def __init__(self, vectors, basis, nxy, roll=(0, 0)):
        self._vectors = np.asarray(vectors)
        self._basis = np.asarray(basis)
        self._nxy = nxy
        self._roll = roll
        self._pos = None
        self._cell = None

    def xy_positions(self):
        if self._pos is None:
            nx, ny = self._nxy
            p = np.mgrid[0:nx, 0:ny].reshape(2, -1).T
            p = (p + self._roll) % [nx, ny]
            q = (p[:, None] + self._basis).reshape(-1, 2)
            self._pos = (q[..., None] * self._vectors).sum(axis=-2)
            self._indices = np.tile(p, (1, self._basis.shape[0])).reshape(-1, 2)
        return self._pos

    def xy_cell(self):
        if self._cell is None:
            nxy = np.asarray(self._nxy).reshape(2, 1)
            self._cell = nxy * self._vectors
        return self._cell

    def select_chunk(self, nxy):
        if self._pos is None:
            self.xy_positions()
        a, b = (self._indices < np.asarray(nxy).reshape(2)).T
        return np.logical_and(a, b)

    def lattice_displacement(self, nxy):
        nx, ny = nxy
        return (np.array([[nx], [ny]]) * self._vectors).sum(axis=0)

    def roll(self, rxy):
        a0, b0 = self._roll
        a, b = rxy
        return LatticePlane(
            self._vectors, self._basis, self._nxy, roll=(a0 + a, b0 + b)
        )


class CubicPlane(LatticePlane):
    def __init__(self, a, nxy, roll=(0, 0)):
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
        super().__init__(uc, b, nxy, roll=roll)


class HexagonalPlane(LatticePlane):
    def __init__(self, a, nxy, roll=(0, 0)):
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
        super().__init__(uc, b, nxy, roll=roll)


class HexagonalPlaneCC(LatticePlane):
    def __init__(self, a, nxy, roll=(0, 0)):
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
        super().__init__(uc, b, nxy, roll=roll)


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


class RepeatedPlane(Plane):
    def __init__(self, plane, repeat):
        self._plane = plane
        self._repeat = np.asarray(repeat)
        self._pos = None

    def xy_positions(self):
        if self._pos is None:
            a, b = self._plane.xy_cell()
            xy = self._plane.xy_positions()
            rx, ry = self._repeat
            self._pos = []
            for i in range(rx):
                for j in range(ry):
                    self._pos.append(xy + i * a + j * b)
            self._pos = np.concatenate(self._pos)
        return self._pos

    def xy_cell(self):
        rep = np.asarray(self._repeat).reshape(2, 1)
        return rep * self._plane.xy_cell()


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

    @property
    def xy_positions(self):
        return self._plane.xy_positions()

    def xyz_positions(self, z):
        return self._plane.xyz_positions(z)

    def subs_atoms(self, atoms):
        return AtomicPlane(self._plane, atoms)

    def subs_chunk(self, nxy, atom):
        assert type(atom) == str
        tags = self._plane.select_chunk(nxy)
        atoms = [atom if t else a for t, a in zip(tags, self.atoms)]
        return self.subs_atoms(atoms)

    def roll(self, rxy):
        return AtomicPlane(self._plane.roll(rxy), self._atoms)

    def repeat(self, rxy):
        if isinstance(self._atoms, str):
            atoms = self._atoms
        else:
            atoms = np.prod(rxy) * self.atoms
        return AtomicPlane(self._plane.repeat(rxy), atoms)

    def __repr__(self):
        return f"AtomicPlane: {Counter(self.atoms)}"

    def as_ase_atoms(self, vacuum=10.0):
        cell_xy = np.c_[self.xy_cell, [0, 0]]
        cell = np.r_[cell_xy, [[0, 0, 2 * vacuum]]]
        atoms = Atoms(
            symbols=self.atoms,
            positions=self.xyz_positions(vacuum),
            cell=cell,
            pbc=True,
        )
        return atoms
