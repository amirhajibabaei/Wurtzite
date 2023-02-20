# +
"""
AgI surface reconstructions.

"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import numpy as np
from numpy.random import RandomState

from wurtzite.atomic_structure import PlaneStacking
from wurtzite.projects.AgI.bulk import LatticeParam, unitcell


@dataclass
class Reconstruction:
    """
    Defined for wurtzite AgI surfaces.
    A surface "Reconstruction" is defined
    by a surface "area" which is (nx, ny)
    of the supercell and re-labeling of
    lattice sites at few layers near the
    surface. Occupied sites are labeled
    by atom types "Ag"/"I" while vacant
    sites are denoted by "X".
    We note that, in addition to
    ideal wurtzite sites in each layer,
    there is an alternative hexagonal lattice
    which we denote as "zincblende sites"
    which play an important role near the
    surface. Therefore for example a 2x2
    layer will be indicated by
        ("Ag Ag Ag X", "X X X X")
    where the first (second) set of symbols
    indicate occupations of the wurtzite
    (zincblende) sites in hexagonal
    sublattices. Then a reconstruction
    is simply defined by a sequence of
    such tuples.

    """

    area: tuple[int, int]
    symbols: tuple[tuple[str, str], ...]
    I_termination: str = "2x1"

    def __post_init__(self):
        assert self.I_termination in ("2x1", "symmetric")
        if self.I_termination == "2x1":
            assert all([n % 2 == 0 for n in self.area])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Reconstruction):
            raise RuntimeError("== is not defined!")
        for a, b in zip(self.area, other.area):
            if a != b:
                return False
        if len(self.symbols) != len(other.symbols):
            return False
        for p, q in zip(self.symbols, other.symbols):
            w1, z1 = p
            w2, z2 = q
            if w1.split() != w2.split() or z1.split() != z2.split():
                return False
        if self.I_termination != other.I_termination:
            return False
        return True

    def get_stacking(
        self, param: LatticeParam | None = None, nz: int = 4, vacuum: float = 20.0
    ) -> PlaneStacking:
        """
        param:
            lattice parametes and number of layers.
        nz:
            number of repeat units along z axis
        vacuum:
            vacuum between periodic images

        if param is None, default params defined
        in "bulk.py" will be used.

        """
        if param is None:
            param = LatticeParam()

        # ideal
        nx, ny = self.area
        stack = unitcell(param).repeat((nx, ny, nz)).with_spacing(-1, vacuum)

        # Ag-terminated reconstruction
        tr = (0.5 * param.a, 0.5 * param.a / sqrt(3))
        for i, (wurtzite, zincblende) in enumerate(self.symbols):
            wz = wurtzite.split()
            zb = zincblende.split()
            plane = stack[i].with_chemical_symbols(wz)
            plane = plane.merge_translation(tr, zb)
            stack = stack.with_plane(i, plane)

        # I-terminated reconstruction
        if self.I_termination == "2x1":
            plane = stack[-1]._plane.repeat(2)  # type: ignore
            plane = plane.with_chemical_symbols("X I I I".split())
            plane = plane.repeat((nx // 2, ny // 2))
            stack = stack.with_plane(-1, plane)
        elif self.I_termination == "symmetric":
            reverse = {"Ag": "I", "I": "Ag", "X": "X"}
            for i, (wurtzite, zincblende) in enumerate(self.symbols):
                wz = [reverse[s] for s in wurtzite.split()]
                zb = [reverse[s] for s in zincblende.split()]
                plane = stack[-i - 1].with_chemical_symbols(wz)
                plane = plane.merge_translation(tr, zb)
                stack = stack.with_plane(-i - 1, plane)
        else:
            raise RuntimeError

        return stack

    def from_stacking(self, stack: PlaneStacking) -> Reconstruction:
        symbols = tuple(
            (
                " ".join(stack[i]._planes[0].get_chemical_symbols()),  # type: ignore
                " ".join(stack[i]._planes[1].get_chemical_symbols()),  # type: ignore
            )
            for i in range(len(self.symbols))
        )
        return Reconstruction(self.area, symbols, self.I_termination)

    def view(self):
        return self.get_stacking().view()

    @staticmethod
    def from_occupations(
        area: tuple[int, int],
        occupations: Sequence[tuple[int, int]],
        rng: RandomState | None = None,
    ) -> Reconstruction:
        nx, ny = area
        n = nx * ny
        symbols = []
        zb = " ".join(n * ["X"])
        for ag, i in occupations:
            assert ag <= n and i <= n
            a = ag * ["Ag"] + (n - ag) * ["X"]
            b = i * ["I"] + (n - i) * ["X"]
            if rng is not None:
                a = rng.permutation(a)
                b = rng.permutation(b)
            symbols.append((" ".join(a), zb))
            symbols.append((" ".join(b), zb))
        re = Reconstruction(area, tuple(symbols))
        return re

    def roll(self, shift: tuple[int, int]) -> Reconstruction:
        def _roll(a):
            b = np.array(a.split()).reshape(self.area)
            c = np.roll(b, shift, axis=(0, 1))
            return " ".join(tuple(c.reshape(-1)))

        symbols = tuple((_roll(w), _roll(z)) for w, z in self.symbols)
        return Reconstruction(self.area, symbols)
