# +
from __future__ import annotations

import abc

from wurtzite.atomic_structure import AtomicStructure
from wurtzite.lammps.fix import Dfield
from wurtzite.lammps.force_field import ForceField
from wurtzite.lammps.structure import FullStyle


class Dynamics(abc.ABC):
    @abc.abstractmethod
    def get_commands(self, group: str) -> list[str]:
        ...


class NVT(Dynamics):
    def __init__(self, dt_fs: float, temp: float, tdamp: int):
        """
        dt_fs:
            timestep in femtoseconds
        temp:
            temperature in Kelvin
        tdamp:
            tdamp in number of steps (LAMMPS recommends ~100)
        """
        self._dt = dt_fs
        self._temp = temp
        self._tdamp = tdamp

    def get_commands(self, group: str) -> list[str]:
        # TODO: for generalization, dt needs conversion!
        fix_id = "nvt_md"
        temp = f"temp {self._temp} {self._temp} $({self._tdamp}*dt)"
        cmd = [
            f"timestep {self._dt}",
            f"compute {group}_temp {group} temp",
            f"fix {fix_id} {group} nvt {temp}",
            f"fix_modify {fix_id} temp {group}_temp",
        ]
        return cmd


class Dump:
    def __init__(self, every_ps: float, file: str):
        self._ps = every_ps
        self._file = file

    def get_commands(self, group: str, types: dict[str, int]) -> list[str]:
        dump_id = "dump_traj"
        columns = "id type element xu yu zu"
        every = f"$({self._ps}*1000/dt)"
        _elements = {b: a for a, b in types.items()}
        elements = " ".join([_elements[e] for e in sorted(_elements)])
        cmd = [
            f"dump {dump_id} {group} custom {every} {self._file} {columns}",
            f"dump_modify {dump_id} element {elements}",
        ]
        return cmd


class Velocities:
    def __init__(self, temp: float, seed: int):
        self._temp = temp
        self._seed = seed

    def get_commands(self, group: str) -> list[str]:
        dist = "gaussian mom yes rot yes"
        cmd = [f"velocity {group} create {self._temp} {self._seed} dist {dist}"]
        return cmd


def run(
    structure: AtomicStructure,
    force_field: ForceField,
    dynamics: Dynamics,
    duration_ns: float,
    *,
    equil_ns: float | None = None,
    velocities: Velocities | None = None,
    dump: Dump | None = None,
    dfield: float | None = None,
):
    # other args
    group = "all"

    # LAMMPS instance
    struc = FullStyle.from_atomic_structure(structure)
    struc.set_forcefield(force_field)
    if dfield is not None:
        struc.set_fix(Dfield(dfield))
    lmp = struc._lmp
    if velocities is not None:
        lmp.commands_list(velocities.get_commands(group))
    lmp.commands_list(dynamics.get_commands(group))

    # equilibration
    if equil_ns is not None:
        steps = f"$({equil_ns}*1e6/dt)"
        lmp.command(f"run {steps}")

    # io
    if dump is not None:
        lmp.commands_list(dump.get_commands(group, struc.get_types()))

    # run
    steps = f"$({duration_ns}*1e6/dt)"
    lmp.command(f"run {steps}")
