# +
from __future__ import annotations

from typing import Sequence

import numpy as np

from wurtzite.lammps.monte_carlo.planes import plane_monte_carlo
from wurtzite.mpi import world
from wurtzite.optimization.genetic_algorithm import GeneticAlgorithm
from wurtzite.projects.AgI.force_field import forcefield
from wurtzite.projects.AgI.reconstruction import Reconstruction


def _monte_carlo(
    re: Reconstruction,
    rng: np.random.RandomState,
    steps: int,
    temp: float,
) -> tuple[Reconstruction, Reconstruction, float]:
    stack = re.get_stacking()
    active_planes = tuple(range(len(re.symbols)))
    fin, opt, energy, rate = plane_monte_carlo(
        stack,
        forcefield,
        active_planes,
        steps=steps,
        temp=temp,
        random_state=rng,
    )
    re_fin = re.from_stacking(fin)
    re_opt = re.from_stacking(opt)
    return re_fin, re_opt, energy


def ga_search(
    area: tuple[int, int],
    occupations: Sequence[tuple[int, int]],
    birthrate: int = 4,
    capacity: int = 16,
    maxstall: int = 4,
    random_seed: int = 57565347,
    filename: str | None = None,
) -> None:
    def minimization(x, rng):
        _, opt, energy = _monte_carlo(
            x,
            rng,
            steps=300,
            temp=100,
        )
        return opt, energy

    def perturbation(x, degree, rng):
        fin, _, _ = _monte_carlo(
            x,
            rng,
            steps=100,
            temp=5000.0 * degree,
        )
        return fin

    def similarity(x, y):
        return x == y

    rng = np.random.RandomState(random_seed)
    algo = GeneticAlgorithm(
        minimization,
        perturbation,
        similarity,
        birthrate=birthrate,
        capacity=capacity,
        random_state=rng,
    )

    re = Reconstruction.from_occupations(area, occupations, rng=rng)
    if world.Get_rank() == 0:
        if filename is None:
            filename = _filename(area, occupations)
        file = open(filename, "w")
        file.write(f"# seed: {re}\n")

    run = algo.irun([re], maxstall=maxstall)
    for i, (sample, degree) in enumerate(run):
        # report
        if world.Get_rank() == 0:
            file.write(f"\nStep: {i} \t count = {len(sample)} \t ({degree = })\n")
            for point in sample:
                file.write(f"{point}\n")
                file.flush()
    if world.Get_rank() == 0:
        file.close()


def _filename(area: tuple[int, int], occupations: Sequence[tuple[int, int]]) -> str:
    suff = "-".join([f"{Ag}Ag{I}I" for Ag, I in occupations])
    file = f"reconst_{area[0]}x{area[1]}_{suff}.txt"
    return file
