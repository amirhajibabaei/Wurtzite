# +
from __future__ import annotations

from io import StringIO

import numpy as np
from ase.calculators.lammps import convert

from wurtzite.pairpot import PairPot


def write_lammps_table(
    pairpots: dict[tuple[str, str], PairPot],
    units: str,
    file: str,
    rmin: float = 0.1,
    rmax: float = 10.0,
    dr: float = 0.01,
    cutoff: float | None = None,
    shift: bool = True,
) -> tuple[int, dict[tuple[str, str], str]]:
    """
    A function for writing tabular potentials to be used with
    "pair_style table" in LAMMPS.

    Args:
        pairpots:
            a dict mapping pairs to potentials
            e.g. {("Ag", "I"): rePRV("Ag", "I"), ...}
        units:
            lammps units to be used for table
            e.g. "metal" or "real", ...
        file:
            filename for table
            e.g. "pot.table"
        rmin, rmax, dr, cutoff:
            self-explanatory distances (in Angstrom units)
            if cutoff is None, it will simply write the potential
            up to rmax.
        shift:
            if True, shifts the potentials to become zero at cutoff

    Returns:
        N:
            N values in lookup
        keys:
            keywords for each pair
            e.g. {("Ag", "I"): "Ag_I", ...}

    """

    with open(file, "w") as of:

        # header
        of.write(f"# UNITS: {units}\n")

        keys = {}
        for pair, pot in pairpots.items():
            # r, e, f
            r = np.arange(rmin, rmax + 1e-8, dr)
            e, f = pot.energy_and_force(r)

            c = -1
            if cutoff is not None:
                c = np.argmin(abs(r - cutoff))

            if shift:
                e -= e[c]
                f -= f[c]

            if cutoff is not None:
                e[c:] = 0
                f[c:] = 0

            # data
            r = convert(r, "distance", "ASE", units)
            e = convert(e, "energy", "ASE", units)
            f = convert(f, "force", "ASE", units)
            N = r.shape[0]
            i = np.arange(1, N + 1)
            data = np.c_[i, r, e, f]

            # write
            key = "_".join(pair)
            keys[pair] = key
            of.write(f"\n{key}")
            of.write(f"\nN {N}\n\n")
            np.savetxt(of, data, fmt=("%10d", "%.18e", "%.18e", "%.18e"))

    return N, keys


def read_lammps_table(file: str) -> dict[str, np.ndarray]:
    """
    It will read a table from file.
    Note that it will convert the table to
    ASE units: A, eV, eV/A
    """

    def _read_blocks(path):
        with open(path) as of:
            blocks = [[]]
            for line in of.readlines():
                if line.strip() == "":
                    blocks.append([])
                else:
                    blocks[-1].append(line)
        blocks = list(filter(lambda b: len(b) > 0, blocks))
        return blocks

    def _get_header_sections(blocks):
        nb = len(blocks)
        start = nb % 2
        sections = [(blocks[i], blocks[i + 1]) for i in range(start, nb, 2)]
        if start == 0:
            header = []
        else:
            header = blocks[0]
        return header, sections

    def _read_section(section, units):
        head, data = section
        assert len(head) == 2, "needs generalization!"
        key = head[0].split()[0]
        key = tuple(key.split("_"))
        N, n, *_ = head[1].split()
        assert N == "N", "needs generalization!"
        val = np.loadtxt(StringIO("".join(data)))
        assert val.shape[0] == int(n)
        # convert back to ASE
        i, r, e, f = val.T
        r = convert(r, "distance", units, "ASE")
        e = convert(e, "energy", units, "ASE")
        f = convert(f, "force", units, "ASE")
        val = np.c_[i, r, e, f]
        return key, val

    def _read_sections(sections, units):
        out = {}
        for sec in sections:
            key, data = _read_section(sec, units)
            out[key] = tuple(data[:, k] for k in range(1, 4))
        return out

    def _read_table(path):
        blocks = _read_blocks(path)
        header, sections = _get_header_sections(blocks)
        for h in header:
            if "UNITS:" in h:
                units = h.split()[-1]
        out = _read_sections(sections, units)
        return out

    return _read_table(file)
