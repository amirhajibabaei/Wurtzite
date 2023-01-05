# +
import abc

import numpy as np
from ase import Atoms

import wurtzite.plane as plane


class Stack:
    def __init__(self, atomic_planes, spacings, vacuum=20.0):
        # assertions
        assert len(atomic_planes) == len(spacings) + 1
        assert all([s >= 0 for s in spacings])
        cells = np.stack([p.xy_cell for p in atomic_planes])
        assert np.allclose(cells - cells[0], 0), "all xy-cells must be identical"

        # init
        self._atomic_planes = atomic_planes
        self._spacings = spacings
        self._vacuum = vacuum

    @property
    def cell(self):
        xy = self._atomic_planes[0].xy_cell
        z = sum(self._spacings) + 2 * self._vacuum
        _xyz = np.c_[xy, [0, 0]]
        cell = np.r_[_xyz, [[0, 0, z]]]
        return cell

    @property
    def z_positions(self):
        z = [self._vacuum]
        for delta in self._spacings:
            z.append(z[-1] + delta)
        return z

    @property
    def positions(self):
        xyz = []
        for p, z in zip(self._atomic_planes, self.z_positions):
            xyz.append(p.get_positions(z))
        return np.concatenate(xyz)

    @property
    def atoms(self):
        symbols = []
        for p in self._atomic_planes:
            symbols.extend(p.atoms)
        return symbols

    def as_ase_atoms(self):
        atoms = Atoms(
            symbols=self.atoms,
            positions=self.positions,
            cell=self.cell,
            pbc=True,
        )
        return atoms

    def __getitem__(self, index):
        return self._atomic_planes[index]

    def __setitem__(self, index, value):
        self._atomic_planes[index] = value


class _WurtZite(Stack):
    def __init__(
        self, cell_a, cell_z1, cell_z2, nxyz, symbols, vacuum=20.0, surf_z1=None
    ):
        nx, ny, nz = nxyz
        A, B = symbols
        a, b = self._hexagonal(cell_a, (nx, ny))
        atomic_planes = nz * [
            plane.AtomicPlane(a, A),
            plane.AtomicPlane(b, B),
            plane.AtomicPlane(b, A),
            plane.AtomicPlane(a, B),
        ]
        spacings = (nz * [cell_z1, cell_z2, cell_z1, cell_z2])[:-1]
        if surf_z1 is not None:
            spacings[0], spacings[-1] = surf_z1
        super().__init__(atomic_planes, spacings, vacuum=vacuum)

    @abc.abstractmethod
    def _hexagonal(self, cell_a, nxy):
        ...


class WurtZiteCC(_WurtZite):
    def _hexagonal(self, cell_a, nxy):
        a = plane.HexagonalPlaneCC(cell_a, nxy)
        b = a.translate([0, cell_a / np.sqrt(3)])
        return a, b


class WurtZite(_WurtZite):
    def _hexagonal(self, cell_a, nxy):
        a = plane.HexagonalPlane(cell_a, nxy)
        b = a.translate([cell_a, cell_a / np.sqrt(3)])
        return a, b
