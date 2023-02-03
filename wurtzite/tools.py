# +
from __future__ import annotations

import itertools
from typing import Any, Iterable, Sequence, Sized

import numpy as np
from np.random import RandomState


def pairings(keys: Iterable[str], self_interaction=True) -> tuple[tuple[str, str], ...]:
    x = tuple(itertools.combinations(keys, 2))
    if self_interaction:
        y = tuple((k, k) for k in keys)
        return (*y, *x)
    else:
        return x


def unchain(chained: Sequence[Any], ref: Sequence[Sized]) -> Sequence[Sized]:
    """
    Inverse of "itertools.chain".

    Example:
        >>> chain = "A B C D E F G".split()
        >>> ref = ((1, 2), (3, 4), (5, 6, 7))
        >>> unchain(chain, ref)
        (('A', 'B'), ('C', 'D'), ('E', 'F', 'G'))

    """
    sizes = tuple(len(x) for x in ref)
    assert sum(sizes) == len(chained)
    sections = tuple(itertools.accumulate(sizes))
    split = np.split(chained, sections)[:-1]
    return tuple(tuple(x) for x in split)


def zip_unchain(
    chained: Sequence[Any], ref: Sequence[Sized]
) -> Sequence[tuple[Sized, Sized]]:
    """
    Example:
        >>> ref = ((1, 2), (3, 4), (5, 6, 7))
        >>> chain = "A B C D E F G".split()
        >>> for a, b in zip_unchain(chain, ref):
        >>>     print(a, b)
        ('A', 'B') (1, 2)
        ('C', 'D') (3, 4)
        ('E', 'F', 'G') (5, 6, 7)

    """
    return tuple(zip(unchain(chained, ref), ref))


def get_random_state(state: int | RandomState) -> RandomState:
    if isinstance(state, int):
        random_state = RandomState(state)
    elif isinstance(state, RandomState):
        random_state = state
    else:
        raise RuntimeError
    return random_state


def test_unchain() -> bool:
    chain = "A B C D E F G".split()
    ref = ((1, 2), (3, 4), (5, 6, 7))
    assert unchain(chain, ref) == (("A", "B"), ("C", "D"), ("E", "F", "G"))
    return True
