# +
from __future__ import annotations

import abc
import io
import tempfile

from mpi4py import MPI


class Communicator(abc.ABC):
    @abc.abstractmethod
    def Get_size(self) -> int:
        ...

    @abc.abstractmethod
    def Get_rank(self) -> int:
        ...

    @abc.abstractmethod
    def Barrier(self) -> None:
        ...


Communicator.register(MPI.Comm)
world = MPI.COMM_WORLD


def strio_to_file(
    strio: io.StringIO, file: "str" | io.TextIOWrapper | None, mode: str = "w"
) -> str:
    if world.Get_rank() == 0:
        if isinstance(file, io.TextIOWrapper):
            file.write(strio.getvalue())
            name = file.name
        elif type(file) == str:
            with open(file, mode) as of:
                of.write(strio.getvalue())
            name = file
        elif file is None:
            tmp = tempfile.NamedTemporaryFile(mode, suffix="wurtzite")
            tmp.write(strio.getvalue())
            name = tmp.name
            # for avoiding tmp deletion, we store in a global list
            _tmpfiles.append(tmp)
        else:
            raise
    else:
        name = None
    name = world.bcast(name, root=0)
    # world.Barrier()
    return name  # type: ignore # mypy can't infer bcast


_tmpfiles: list[tempfile._TemporaryFileWrapper] = []
