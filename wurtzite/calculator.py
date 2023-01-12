# +
from ase.calculators.calculator import Calculator, all_changes

from wurtzite.lammps import LAMMPS
from wurtzite.pairpot import PairPot


class PairStyleCalculator(Calculator):

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        pairpots: dict[tuple[str, str], PairPot],
        cutoff: float,
        charges: dict[str, float],
        kspace_style: str = "pppm 1e-4",
        table_dr: float = 0.01,
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self._pairpots = pairpots
        self._cutoff = cutoff
        self._charges = charges
        self._table_dr = table_dr
        self._kspace_style = kspace_style

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if "numbers" in system_changes:
            self._lmp = LAMMPS(atoms)
            self._lmp.set_pair_coeffs(
                self._pairpots,
                self._cutoff,
                kspace_style=self._kspace_style,
                dr=self._table_dr,
            )
            self._lmp.set_charges(self._charges)
        else:
            self._lmp.update_atoms(atoms)
        self.results["energy"] = self._lmp.get_potential_energy(units="ASE")
