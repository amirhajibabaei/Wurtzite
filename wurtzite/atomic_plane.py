# +
from __future__ import annotations

import abc
from collections.abc import Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers


class AtomicPlane(abc.ABC):

    # For children:

    @abc.abstractmethod
    def get_xy_positions(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def get_xy_cell(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def get_chemical_symbols(self) -> Sequence[str]:
        ...

    # Derived:

    def get_xyz_positions(self, z: float) -> np.ndarray:
        xy = self.get_xy_positions()
        z = np.full(xy.shape[0], z)
        return np.c_[xy, z]

    def get_area(self) -> float:
        return abs(np.linalg.det(self.get_xy_cell()))

    def __len__(self) -> int:
        return len(self.get_xy_positions())

    def with_chemical_symbols(self, symbols: str | Sequence[str]) -> GenericPlane:
        return GenericPlane(self.get_xy_positions(), self.get_xy_cell(), symbols)

    def repeat(self, repeat) -> Repetition:
        return Repetition(self, repeat)

    def translate(self, tr: tuple[float, float]) -> Translation:
        return Translation(self, tr)

    def merge(self, other: AtomicPlane) -> Merge:
        return Merge(self, other)

    def permute(self, perm: Sequence[int]) -> GenericPlane:
        symbols = self.get_chemical_symbols()
        new = [symbols[i] for i in perm]
        return self.with_chemical_symbols(new)

    def to_ase_atoms(self, z=10.0) -> Atoms:
        _cell = np.c_[self.get_xy_cell(), [0, 0]]
        cell = np.r_[_cell, [[0, 0, 2 * z]]]
        atoms = Atoms(
            symbols=self.get_chemical_symbols(),
            positions=self.get_xyz_positions(z),
            cell=cell,
            pbc=True,
        )
        return atoms


class GenericPlane(AtomicPlane):
    def __init__(
        self,
        xy: np.ndarray,
        xy_cell: np.ndarray,
        symbols: str | Sequence[str],
        scaled_positions: bool = False,
    ):
        if scaled_positions:
            xy = (xy[..., None] * xy_cell).sum(axis=-2)
        if type(symbols) == str:
            symbols = xy.shape[0] * [symbols]
        for symbol in symbols:
            assert symbol in atomic_numbers
        self._xy = xy
        self._symbols = tuple(symbols)
        self._xy_cell = xy_cell

    # From parents:

    def get_xy_positions(self) -> np.ndarray:
        return self._xy

    def get_xy_cell(self) -> np.ndarray:
        return self._xy_cell

    def get_chemical_symbols(self) -> Sequence[str]:
        return self._symbols


class Merge(AtomicPlane):
    def __init__(self, plane_1: AtomicPlane, plane_2: AtomicPlane):
        assert np.allclose(plane_1.get_xy_cell(), plane_2.get_xy_cell())
        self._plane_1 = plane_1
        self._plane_2 = plane_2

    # From parents:

    def get_xy_cell(self) -> np.ndarray:
        return self._plane_1.get_xy_cell()

    def get_xy_positions(self) -> np.ndarray:
        xy = np.concatenate(
            [self._plane_1.get_xy_positions(), self._plane_2.get_xy_positions()]
        )
        return xy

    def get_chemical_symbols(self) -> Sequence[str]:
        symbols = (
            *self._plane_1.get_chemical_symbols(),
            *self._plane_2.get_chemical_symbols(),
        )
        return symbols


class _PlaneMixin:
    """
    All subclasses set self._plane attribute upon init.
    A few or all of the following methods can be overwritten by children.
    """

    _plane: AtomicPlane

    def get_xy_positions(self) -> np.ndarray:
        return self._plane.get_xy_positions()

    def get_xy_cell(self) -> np.ndarray:
        return self._plane.get_xy_cell()

    def get_chemical_symbols(self) -> Sequence[str]:
        return self._plane.get_chemical_symbols()


class CubicPlane(_PlaneMixin, AtomicPlane):
    def __init__(self, a: float, symbols: str | Sequence[str]):
        xy_cell = (
            np.array(
                [
                    [1, 0],  #
                    [0, 1],  #
                ]
            )
            * a
        )
        xy = np.array([[0, 0]])
        self._plane = GenericPlane(xy, xy_cell, symbols, scaled_positions=True)


class HexagonalPlane(_PlaneMixin, AtomicPlane):
    def __init__(self, a: float, symbols: str | Sequence[str]):
        xy_cell = (
            np.array(
                [
                    [1, 0],  #
                    [1 / 2, np.sqrt(3) / 2],  #
                ]
            )
            * a
        )
        xy = np.array([[0, 0]])
        self._plane = GenericPlane(xy, xy_cell, symbols, scaled_positions=True)


class HexagonalPlane2(_PlaneMixin, AtomicPlane):
    def __init__(self, a: float, symbols: str | Sequence[str]):
        xy_cell = (
            np.array(
                [
                    [1, 0],  #
                    [0, np.sqrt(3)],  #
                ]
            )
            * a
        )
        xy = np.array([[0, 0], [0.5, 0.5]])
        self._plane = GenericPlane(xy, xy_cell, symbols, scaled_positions=True)


class Translation(_PlaneMixin, AtomicPlane):
    def __init__(self, plane: AtomicPlane, tr: tuple[float, float]):
        self._plane = plane
        self._tr = tr

    # Overloads:

    def get_xy_positions(self) -> np.ndarray:
        return self._plane.get_xy_positions() + self._tr


def _repeat_tuple(repeat: int | tuple[int, int]) -> tuple[int, int]:
    if type(repeat) == int:
        return (repeat, repeat)
    elif type(repeat) == tuple:
        return repeat
    else:
        raise RuntimeError


class Repetition(_PlaneMixin, AtomicPlane):
    def __init__(self, plane: AtomicPlane, repeat: int | tuple[int, int]):
        self._plane = plane
        self._repeat = _repeat_tuple(repeat)

    # From parents:
    def get_xy_positions(self) -> np.ndarray:
        a, b = self._plane.get_xy_cell()
        _xy = self._plane.get_xy_positions()
        rx, ry = self._repeat
        xy = []
        for i in range(rx):
            for j in range(ry):
                xy.append(_xy + i * a + j * b)
        return np.concatenate(xy)

    def get_xy_cell(self) -> np.ndarray:
        repeat = np.asarray(self._repeat).reshape(2, 1)
        return repeat * self._plane.get_xy_cell()

    def get_chemical_symbols(self) -> Sequence[str]:
        return self._plane.get_chemical_symbols() * np.prod(self._repeat)

    # Overloads:

    def repeat(self, repeat: int | tuple[int, int]) -> Repetition:
        a1, b1 = _repeat_tuple(repeat)
        a2, b2 = self._repeat
        return Repetition(self._plane, (a1 * a2, b1 * b2))

    # Derived:

    def mask_of_block(
        self, nxy: int | tuple[int, int], *, origin: int | tuple[int, int] = 0
    ) -> Sequence[bool]:
        rx, ry = self._repeat
        nx, ny = _repeat_tuple(nxy)
        ox, oy = _repeat_tuple(origin)
        n = len(self._plane)
        block = []
        for i in range(rx):
            ii = i - ox
            for j in range(ry):
                jj = j - oy
                if ii >= 0 and ii < nx and jj >= 0 and jj < ny:
                    block.extend(n * [True])
                else:
                    block.extend(n * [False])
        return block

    def with_block_symbols(
        self,
        nxy: int | tuple[int, int],
        symbol: str,
        *,
        origin: int | tuple[int, int] = 0,
    ) -> GenericPlane:
        mask = self.mask_of_block(nxy, origin=origin)
        symbols = tuple(
            symbol if m else s for m, s in zip(mask, self.get_chemical_symbols())
        )
        return GenericPlane(self.get_xy_positions(), self.get_xy_cell(), symbols)

    def __repr__(self):
        nx, ny = self._repeat
        return f"Repetition({nx}, {ny})"


def test_planes() -> bool:
    c = CubicPlane(1.0, "H")
    h1 = HexagonalPlane(1.0, "H")
    h2 = HexagonalPlane2(1.0, "H")
    for p in [c, h1, h2]:
        tr = p.translate((0.0, 0.0))
        assert np.allclose(p.get_xy_positions(), tr.get_xy_positions())
        p22 = p.repeat((2, 2))
        assert len(p22) == 4 * len(p)
        p1 = p.with_chemical_symbols("He")
        p2 = p.with_chemical_symbols(len(p) * ["He"])
        assert p1.get_chemical_symbols() == p2.get_chemical_symbols()
    return True


if __name__ == "__main__":
    test_planes()
