# +
import abc
from typing import Sequence

from ase.calculators.lammps import convert

from wurtzite.lammps._backend import _qqr2e


class Fix(abc.ABC):
    @abc.abstractmethod
    def get_commands(self, units: str) -> Sequence[str]:
        ...


class Dfield(Fix):
    def __init__(
        self,
        dfield: float,
        direction: str = "z",
        group: str = "all",
        modify_energy: str = "yes",
        modify_virial: str = "yes",
        fixid: str = "dfield",
    ):
        """
        dfield units: e/A^2

        """
        assert direction in "xyz"
        assert modify_energy in ("yes", "no")
        assert modify_virial in ("yes", "no")
        self._dfield = dfield
        self._direction = direction
        self._group = group
        self._energy = modify_energy
        self._virial = modify_virial
        self._fixid = fixid

    def get_commands(self, units: str) -> Sequence[str]:
        # fixid
        fid = self._fixid

        # dfield
        e = convert(1.0, "charge", "ASE", units)
        A = convert(1.0, "distance", "ASE", units)
        u = e / A**2
        dfield = self._dfield / u  # TODO: check

        # directions
        ru = f"{self._direction}u"
        if ru == "xu":
            force = f"v_{fid}_F 0 0"
        elif ru == "yu":
            force = f"0 v_{fid}_F 0"
        elif ru == "zu":
            force = f"0 0 v_{fid}_F"
        else:
            raise RuntimeError("... in Dfield!")

        commands = [
            f"group      {fid}_g   union   {self._group}",
            f"variable   {fid}_D   equal   {dfield}",
            f"variable   {fid}_C   equal   {_qqr2e[units]}",
            f"variable   {fid}_eps equal   1/(4*PI*v_{fid}_C)",
            f"compute    {fid}_r   {fid}_g property/atom {ru}",
            f"variable   {fid}_p   atom    q*c_{fid}_r/vol",
            f"compute    {fid}_P   {fid}_g reduce sum v_{fid}_p",
            f"variable   {fid}_E   equal   v_{fid}_D-c_{fid}_P",
            f"variable   {fid}_F   atom    q*v_{fid}_E/v_{fid}_eps",
            f"variable   {fid}_u   atom    vol*v_{fid}_E^2/(2*atoms*v_{fid}_eps)",
            f"fix        {fid}     {fid}_g addforce {force} energy v_{fid}_u",
            # f"compute    {fid}_U   {fid}_g reduce sum v_{fid}_u"
            f"variable   {fid}_U   equal   vol*v_{fid}_E^2/(2*v_{fid}_eps)",
            f"fix_modify {fid}     energy {self._energy} virial  {self._virial}",
        ]
        return commands
