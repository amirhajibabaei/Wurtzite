# +
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

from numpy.random import RandomState

from wurtzite.tools import expit, get_random_state, logit


@dataclass
class Point:
    x: Any
    y: float


class GeneticAlgorithm:
    """
    A variation of genetic algorithm for searching
    the global minimum of a function with many local
    minima.
    Three functions are required for the algorithm:

    1) A local minimizer with the following signature:

        def minimization(x: Any, rng: RandomState) -> Any, float:
            ...
            return x_min, y_min

       A random state "rng" is given which should be
       used for drawing random numbers (if needed).
       Global rng is required for mpi/parallel simulations.

    2) A perturbation function for a random displacements
       of local minima where the degree of perturbation,
       a number in the range (0, 1), is controled by the
       "degree" argument.
       When degree -> 1, the perturbation should be strong
       enough to throw the input out of any local minimum.
       Also when degree -> 0 close enough, a perturbation
       followed by a local minimization should reach the
       same point.
       A random state "rng" is also given which should be
       used for drawing random numbers.
       The function signature is:

        def perturbation(x: Any, degree: float, rng: RandomState) -> Any:
            ...
            return x_rand

    3) A simularity function which returns True if two
       local mimimums are equivalent:

        def similarity(x_1: Any, x_2: Any) -> bool:
            ...
            return True or False

    """

    def __init__(
        self,
        minimization: Callable[[Any, RandomState], tuple[Any, float]],
        perturbation: Callable[[Any, float, RandomState], Any],
        similarity: Callable[[Any, Any], bool],
        *,
        birthrate: int = 1,
        capacity: int = 1,
        random_state: int | RandomState = 5631,
    ):
        self._minimization = minimization
        self._perturbation = perturbation
        self._similarity = similarity
        self._birthrate = birthrate
        self._capacity = capacity
        self._rand = get_random_state(random_state)

    def _step(
        self, sample: Sequence[Any] | Sequence[Point], degree: float
    ) -> tuple[Sequence[Point], float, int]:
        def _init(x: Any | Point) -> Point:
            if isinstance(x, Point):
                return x
            else:
                xmin, ymin = self._minimization(x, self._rand)
                return Point(xmin, ymin)

        # population:
        parents: list[Point] = []
        children: list[Point] = []
        for _p in sample:
            p = _init(_p)
            parents.append(p)
            for _ in range(self._birthrate):
                _qx = self._perturbation(p.x, degree, self._rand)
                qx, qy = self._minimization(_qx, self._rand)
                q = Point(qx, qy)
                if not self._similarity(q.x, p.x):
                    children.append(q)
        alpha = len(children) / (len(parents) * self._birthrate)

        # selection:
        old_and_new = sorted([*parents, *children], key=lambda q: q.y)
        selection: list[Point] = []
        for a in old_and_new:
            # no duplicates
            new = True
            for u in selection:
                if self._similarity(u.x, a.x):
                    new = False
                    break
            if new:
                selection.append(a)
                if len(selection) >= self._capacity:
                    break

        # beta: number of new minimums
        beta = len(selection)
        for s in selection:
            for p in parents:
                if self._similarity(p.x, s.x):
                    beta -= 1
                    break

        return selection, alpha, beta

    def irun(
        self,
        sample: Sequence[Any] | Sequence[Point],
        maxstall: int = 10,
        maxsteps: int = 100,
        degree: float = 0.1,
    ) -> Iterator[tuple[Sequence[Point], float]]:

        stalled = 0
        for _ in range(maxsteps):

            if stalled > maxstall:
                break

            sample, alpha, beta = self._step(sample, degree)

            # tune randomness degree
            _degree = self._rand.uniform(-0.1, 0.1)
            if alpha < 0.3:
                _degree += 0.9
            elif alpha > 0.7:
                _degree -= 0.9
            degree = expit(logit(degree) + _degree)

            if beta > 0:
                stalled = 0
                continue

            # count as stall only if alpha > 0
            if alpha > 0:
                stalled += 1

            yield sample, degree


def test_GeneticAlgorithm() -> bool:
    import functools

    import numpy as np
    from scipy.optimize import minimize

    def objective(x: np.ndarray) -> float:
        """
        "Rastrigin" function with a global minimum at 0

        """
        x = np.asarray(x).reshape(-1)
        n = len(x)
        A = 10
        y = A * n + (x**2 - A * np.cos(2 * np.pi * x)).sum()
        return y

    def minimization(x: np.ndarray, rng: RandomState) -> tuple[np.ndarray, float]:
        res = minimize(objective, x)
        return res.x, res.fun

    def perturbation(x: np.ndarray, degree: float, rng: RandomState) -> np.ndarray:
        x = np.asarray(x).reshape(-1)
        return x + rng.uniform(-1.0, 1.0, size=len(x)) * degree

    similarity = functools.partial(np.allclose, atol=0.01)

    self = GeneticAlgorithm(
        minimization, perturbation, similarity, birthrate=4, capacity=4
    )

    run = self.irun([np.array(2.7)])
    for i, (sample, degree) in enumerate(run):
        pass
    assert sample[0].y < 1e-8

    return True


if __name__ == "__main__":
    test_GeneticAlgorithm()
