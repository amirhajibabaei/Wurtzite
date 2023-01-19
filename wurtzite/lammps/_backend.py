# +
from __future__ import annotations

import ctypes
from typing import Sequence

import numpy as np
from ase.calculators.lammps import convert
from ase.data import atomic_masses, atomic_numbers
from ase.geometry import wrap_positions
from lammps import lammps as _lammps

from wurtzite.rotations import PrismRotation


class lammps(_lammps):
    _history: list[str] | None = None
    _ignore: list[str] = ["#", "change_box", "run"]

    def command(self, cmd: str) -> None:
        super().command(cmd)
        self._memorize([cmd])

    def commands_list(self, cmdlist: Sequence[str]) -> None:
        super().commands_list(cmdlist)
        self._memorize(cmdlist)

    def commands_string(self, multicmd: str) -> None:
        super().commands_string(multicmd)
        self._memorize(multicmd.split("\n"))

    def _memorize(self, cmdlist: Sequence[str]) -> None:
        if self._history is None:
            self._history = []
        for _cmd in cmdlist:
            cmd = _cmd.strip()
            ignore = any([cmd.startswith(ignore) for ignore in self._ignore])
            if not ignore:
                self._history.append(cmd)

    def history(self) -> str:
        if self._history is None:
            return ""
        else:
            return "\n".join(self._history)


def _create_lammps(
    positions: np.ndarray,
    cell: np.ndarray,
    symbols: Sequence[str],
    pbc: bool | tuple[bool, bool, bool],
    units: str,
    log: str = "none",
    screen: str = "none",
) -> tuple[lammps, dict[str, int]]:

    # process args
    assert positions.shape[1] == 3
    assert cell.shape == (3, 3)
    assert len(symbols) == positions.shape[0]
    pbc = _pbc_tuple(pbc)
    types = {symbol: index + 1 for index, symbol in enumerate(np.unique(symbols))}

    # create the LAMMPS object
    lmp = lammps(cmdargs=f"-log {log} -screen {screen}".split())
    lmp.commands_list(
        [
            f"units {units}",  # units
            "atom_style full",  # atom style
            _pbc_to_str(pbc),  # boundary
        ]
    )

    # create box and atoms
    prism, xyz = _get_prism_coordinates(pbc, cell, positions, units)
    region_id = "cell"
    prism = "0 {} 0 {} 0 {} {} {} {}".format(*prism)
    lmp.command(f"region {region_id} prism {prism} units box")
    lmp.command(f"create_box {len(types)} {region_id}")
    lmp.create_atoms(
        n=xyz.shape[0],
        id=list(range(1, xyz.shape[0] + 1)),
        type=list(map(types.get, symbols)),
        x=xyz.reshape(-1).tolist(),
        v=None,
        image=None,
        shrinkexceed=False,
    )

    # masses
    for symbol, type_ in types.items():
        number = atomic_numbers[symbol]
        mass = convert(atomic_masses[number], "mass", "ASE", units)
        lmp.command(f"mass {type_} {mass}")

    return lmp, types


def _update_lammps_positions(
    lmp: lammps,
    pbc: tuple[bool, bool, bool],
    cell: np.ndarray,
    positions: np.ndarray,
    units: str,
) -> None:

    prism, positions = _get_prism_coordinates(pbc, cell, positions, units)

    # update box
    # box = " ".join([f"{a} final 0 {b}" for a, b in zip(["x", "y", "z"], prism[:3])])
    # tilt = " ".join([f"{a} final {b}" for a, b in zip(["xy", "xz", "yz"], prism[3:])])
    # lmp.command(f"change_box all {box} {tilt} {_pbc_to_str(pbc)}")

    # update positions
    lmp.scatter_atoms("x", 1, 3, _array_c_ptr(positions.reshape(-1)))


def _update_lammps_cell(
    lmp: lammps,
    cell: np.ndarray,
    units: str,
) -> None:

    rotation = PrismRotation(cell)
    rot_cell = rotation(cell)
    con_cell = convert(rot_cell, "distance", "ASE", units)
    assert np.allclose(con_cell.flat[[1, 2, 5]], 0)
    prism = con_cell.flat[[0, 4, 8, 3, 6, 7]]

    # update box
    box = " ".join([f"{a} final 0 {b}" for a, b in zip(["x", "y", "z"], prism[:3])])
    tilt = " ".join([f"{a} final {b}" for a, b in zip(["xy", "xz", "yz"], prism[3:])])
    lmp.command(f"change_box all {box} {tilt}")


def _update_lammps_pbc(lmp, pbc: tuple[bool, bool, bool]) -> None:
    lmp.command(f"change_box all {_pbc_to_str(pbc)}")


def _get_prism_coordinates(
    pbc: tuple[bool, bool, bool],
    cell: np.ndarray,
    positions: np.ndarray,
    units: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    TODO: don't wrap positions, use image numbers instead.
    """

    # rotate coordinates to LAMMPS "prism" style
    rotation = PrismRotation(cell)
    rot_cell = rotation(cell)
    rot_positions = rotation(positions)

    # wrap
    rot_positions = wrap_positions(rot_positions, rot_cell, pbc)

    # convert units
    con_cell = convert(rot_cell, "distance", "ASE", units)
    con_positions = convert(rot_positions, "distance", "ASE", units)

    # prism
    assert np.allclose(con_cell.flat[[1, 2, 5]], 0)
    prism = con_cell.flat[[0, 4, 8, 3, 6, 7]]
    return prism, con_positions


def _array_c_ptr(arr):
    ptr = ctypes.POINTER(ctypes.c_double)
    return arr.astype(np.float64).ctypes.data_as(ptr)


def _pbc_to_str(pbc: tuple[bool, bool, bool]) -> str:
    return "boundary " + " ".join(["p" if b else "f" for b in pbc])


def _pbc_tuple(pbc: bool | tuple[bool, bool, bool]) -> tuple[bool, bool, bool]:
    if type(pbc) == bool:
        return (pbc, pbc, pbc)
    elif isinstance(pbc, Sequence):
        return pbc
    if isinstance(pbc, np.ndarray):
        assert len(pbc) == 3
        return (pbc[0], pbc[1], pbc[2])
    else:
        raise RuntimeError(f"got {pbc} for pbc!")


_qqr2e = {
    "lj": 1.0,
    "real": 332.06371,
    "metal": 14.399645,
    "si": 8987600000.0,
    "cgs": 1.0,
    "electron": 1.0,
    "micro": 8987556.0,
    "nano": 230.7078669,
}
