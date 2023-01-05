# +
import abc
from io import StringIO

import numpy as np
from ase.calculators.lammps import convert


class PairPot(abc.ABC):
    """
    ASE units are assumed
        distance: A (Angstrom)
        energy: eV
        force: eV/A
    """

    @abc.abstractmethod
    def energy_and_force(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        input: r (A)
        returns: e (eV), f (ev/A)
        """
        ...


class _rePRV(PairPot):
    """
    Parrinello–Rahman–Vashista
    """

    def __init__(self, eta, H, P, C):
        """
              |  eta      H      P       C
        ----------------------------------
        Ag-Ag |  11     0.16      0      0
        Ag-I  |   9   1310.0   14.9      0
        I-I   |   7   5328.0   29.8   84.5

        """
        self._params = eta, H, P, C

    def energy_and_force(self, r):
        eta, H, P, C = self._params
        e = H / r**eta - C / r**6 - P / r**4
        f = eta * H / r ** (eta + 1) - 6 * C / r**7 - 4 * P / r**5
        return e, f


class rePRV(_rePRV):

    parameters = {
        ("Ag", "Ag"): (11, 0.16, 0, 0),
        ("Ag", "I"): (9, 1310.0, 14.9, 0),
        ("I", "I"): (7, 5328.0, 29.8, 84.5),
    }

    def __init__(self, pair):
        a, b = pair
        try:
            par = self.parameters[(a, b)]
        except KeyError:
            par = self.parameters[(b, a)]
        except KeyError:
            raise RuntimeError(f"rePRV is not defined for {a}, {b}")
        super().__init__(*par)


def write_table(
    pairpots,
    units,
    file,
    rmin=0.1,
    rmax=10.0,
    dr=0.01,
    cutoff=None,
    shift=True,
):
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
            r = convert(r, "distance", "ASE", units)
            e = convert(e, "energy", "ASE", units)
            f = convert(f, "force", "ASE", units)

            # write
            N = r.shape[0]
            i = np.arange(1, N + 1)
            data = np.c_[i, r, e, f]
            key = "_".join(pair)
            keys[pair] = (key, N)
            of.write(f"\n{key}")
            of.write(f"\nN {N}\n\n")
            np.savetxt(of, data, fmt=("%10d", "%.18e", "%.18e", "%.18e"))
    return keys


def read_table(file):
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
