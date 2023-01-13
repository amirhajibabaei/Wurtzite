# +
from ase.calculators.calculator import Calculator, all_changes

from wurtzite.lammps.forcefield import ForceField
from wurtzite.lammps.structure import FullStyle


class PairStyleCalculator(Calculator):

    implemented_properties = ["energy", "forces"]

    def __init__(self, ff: ForceField, **kwargs):
        Calculator.__init__(self, **kwargs)
        self._ff = ff

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if "numbers" in system_changes:
            self._lmp = FullStyle.from_atomic_structure(self.atoms)
            self._lmp.apply_forcefield_(self._ff)
        else:
            self._lmp.set_pbc(self.atoms.pbc)
            self._lmp.set_cell(self.atoms.cell)
            self._lmp.set_positions(self.atoms.positions)
        self.results["energy"] = self._lmp.get_potential_energy(units="ASE")
