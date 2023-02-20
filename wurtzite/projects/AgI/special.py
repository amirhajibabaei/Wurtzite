# +
from wurtzite.projects.AgI.reconstruction import Reconstruction

__all__ = [
    "vacancy",
    "mosaic",
    "honeycomb",
    "ladder",
    "ladder_staggered",
    "ladder_broken",
    "varray",
    "zigzag",
    "zigzag_flip",
]

# simplest possible reconstruction: array of vacancies
vacancy = Reconstruction(
    area=(2, 2), symbols=(("X Ag Ag Ag", "X X X X"), ("I I I I", "X X X X"))
)

mosaic = Reconstruction(
    area=(4, 4),
    symbols=(
        (
            "Ag Ag Ag Ag Ag X X X Ag X X Ag Ag X Ag X",
            "X X X X X Ag Ag X X Ag X X X X X X",
        ),
        ("I I I I I I I I I I I I I I I I", "X X X X X X X X X X X X X X X X"),
    ),
)

# Ag ions form a honeycomb lattice
honeycomb = Reconstruction(
    area=(2, 2), symbols=(("X Ag X X", "X X Ag X"), ("I X I I", "X X X X"))
)

# cubic arrangement of Ag atoms forms ladder-like structures
ladder = Reconstruction(
    area=(4, 4),
    symbols=(
        ("X X X X Ag Ag Ag Ag X X X X X X X X", "X X X X X X X X Ag Ag Ag Ag X X X X"),
        ("I I I I I I I I I I I I X X X X", "X X X X X X X X X X X X X X X X"),
    ),
)
ladder_staggered = Reconstruction(
    area=(6, 6),
    symbols=(
        (
            "X X X Ag X X Ag X Ag X X X X X X X X Ag "
            "X X Ag X Ag X X Ag X X X X Ag X X X Ag X",
            "X Ag X X X Ag X X X X Ag X X Ag X Ag X X "
            "Ag X X X X X X X X Ag X Ag X X Ag X X X",
        ),
        (
            "I I I X I I X I X I I I I I I I I X I I X I X I I X I I I I X I I I X I",
            "X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X",
        ),
    ),
)
ladder_broken = Reconstruction(
    area=(4, 4),
    symbols=(
        ("X X X X X X Ag Ag X X X X X X X X", "X X X X X X X X X Ag Ag X X X X X"),
        ("X I I I X I I X I I I X X X X X", "X X X X X X X X X X X X X X X X"),
    ),
)


#
varray = Reconstruction(
    area=(2, 2), symbols=(("X X X Ag", "X X X X"), ("X I I X", "X X X X"))
)
zigzag = Reconstruction(
    area=(4, 4),
    symbols=(
        ("X X Ag X X X X X X X Ag X X X X X", "X X X X X Ag X X X X X X X Ag X X"),
        ("X I X X I I I X X I X X I I I X", "X X X X X X X X X X X X X X X X"),
    ),
)
zigzag_flip = Reconstruction(
    area=(4, 4),
    symbols=(
        ("X X Ag X X X X X X X Ag X X X Ag X", "X X X X X Ag X X X X X X X X X X"),
        ("X I X X I I I X X I I X X I I X", "X X X X X X X X X X X X X X X X"),
    ),
)
