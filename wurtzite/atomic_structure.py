# +
from __future__ import annotations

import abc
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from wurtzite.atomic_plane import AtomicPlane, HexagonalPlane, HexagonalPlane2


class AtomicStructure(abc.ABC):

    # For children:

    @abc.abstractmethod
    def get_pbc(self) -> tuple[bool, bool, bool]:
        ...

    @abc.abstractmethod
    def get_cell(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def get_positions(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def get_chemical_symbols(self) -> Sequence[str]:
        ...

    # Derived:

    def to_ase_atoms(self) -> Atoms:
        atoms = Atoms(
            symbols=self.get_chemical_symbols(),
            positions=self.get_positions(),
            cell=self.get_cell(),
            pbc=self.get_pbc(),
        )
        return atoms

    def get_atomic_numbers(self) -> Sequence[int]:
        return tuple(atomic_numbers[s] for s in self.get_chemical_symbols())

    def __len__(self) -> int:
        return len(self.get_chemical_symbols())

    def get_volume(self) -> float:
        return abs(np.linalg.det(self.get_cell()))


class DynamicStructure(AtomicStructure):

    # For children:

    @abc.abstractmethod
    def set_positions(self, positions: np.ndarray) -> None:
        ...

    @abc.abstractmethod
    def set_cell(self, cell: np.ndarray) -> None:
        ...

    @abc.abstractmethod
    def set_pbc(self, pbc: bool | tuple[bool, bool, bool]) -> None:
        ...


DynamicStructure.register(Atoms)


class PlaneStacking(AtomicStructure):

    # From parents:

    def get_pbc(self) -> tuple[bool, bool, bool]:
        return 3 * (True,)

    def get_cell(self) -> np.ndarray:
        xy = self.get_planes()[0].get_xy_cell()
        z = sum(self.get_spacings())
        _xyz = np.c_[xy, [0, 0]]
        cell = np.r_[_xyz, [[0, 0, z]]]
        return cell

    def get_positions(self) -> np.ndarray:
        xyz = []
        for plane, z in zip(self.get_planes(), self._get_z()):
            xyz.append(plane.get_xyz_positions(z))
        return np.concatenate(xyz)

    def get_chemical_symbols(self) -> Sequence[str]:
        symbols: list[str] = []
        for plane in self.get_planes():
            symbols.extend(plane.get_chemical_symbols())
        return symbols

    # For children:

    @abc.abstractmethod
    def get_planes(self) -> Sequence[AtomicPlane]:
        ...

    @abc.abstractmethod
    def get_spacings(self) -> Sequence[float]:
        ...

    # Derived:

    def get_num_planes(self) -> int:
        return len(self.get_planes())

    def get_plane(self, index: int) -> AtomicPlane:
        return self.get_planes()[index]

    def with_plane(self, index: int, plane: AtomicPlane) -> GenericStacking:
        _planes = self.get_planes()
        index = index % len(_planes)
        planes = tuple(plane if i == index else p for i, p in enumerate(_planes))
        assert all_xy_cells_are_identical(planes)
        return GenericStacking(planes, self.get_spacings())

    def with_spacing(self, index: int, spacing: float) -> GenericStacking:
        _spacings = self.get_spacings()
        index = index % len(_spacings)
        spacings = tuple(spacing if i == index else s for i, s in enumerate(_spacings))
        return GenericStacking(self.get_planes(), spacings)

    def _get_z(self) -> list[float]:
        z = [0.0]
        for delta in self.get_spacings():
            z.append(z[-1] + delta)
        return z

    def get_surface_area(self) -> float:
        return self.get_plane(0).get_area()

    def repeat(self, repeat: int | tuple[int, int, int]) -> GenericStacking:
        if type(repeat) == int:
            nx = ny = nz = repeat
        elif isinstance(repeat, tuple):
            nx, ny, nz = repeat
        planes = nz * [plane.repeat((nx, ny)) for plane in self.get_planes()]
        spacings = nz * list(self.get_spacings())
        return GenericStacking(planes, spacings)

    def index_range(self, plane_index: int) -> tuple[int, int]:
        planes = self.get_planes()
        index = plane_index % len(planes)
        i = sum([len(p) for p in planes[:index]])
        j = i + len(planes[index])
        return i, j


class _StackingMixin:
    _planes: Sequence[AtomicPlane]
    _spacings: Sequence[float]

    def get_planes(self) -> Sequence[AtomicPlane]:
        return self._planes

    def get_spacings(self) -> Sequence[float]:
        return self._spacings


class GenericStacking(_StackingMixin, PlaneStacking):
    def __init__(self, planes: Sequence[AtomicPlane], spacings: Sequence[float]):
        assert len(planes) == len(spacings)
        assert all_xy_cells_are_identical(planes)
        self._planes = tuple(planes)
        self._spacings = tuple(spacings)


def all_xy_cells_are_identical(planes) -> bool:
    cells = np.stack([plane.get_xy_cell() for plane in planes])
    return np.allclose(cells - cells[0], 0)


class WurtZite(_StackingMixin, PlaneStacking):
    def __init__(
        self,
        a: float,
        z1: float,
        z2: float,
        symbols: tuple[str, str],
    ):
        A, B = symbols
        x = HexagonalPlane(a, "X")
        y = x.translate((a, a / np.sqrt(3)))
        self._planes = [
            x.with_chemical_symbols(A),
            y.with_chemical_symbols(B),
            y.with_chemical_symbols(A),
            x.with_chemical_symbols(B),
        ]
        self._spacings = 2 * [z1, z2]


class WurtZite2(_StackingMixin, PlaneStacking):
    def __init__(
        self,
        a: float,
        z1: float,
        z2: float,
        symbols: tuple[str, str],
    ):
        A, B = symbols
        x = HexagonalPlane2(a, "X")
        y = x.translate((0, a / np.sqrt(3)))
        self._planes = [
            x.with_chemical_symbols(A),
            y.with_chemical_symbols(B),
            y.with_chemical_symbols(A),
            x.with_chemical_symbols(B),
        ]
        self._spacings = 2 * [z1, z2]
