# +
import abc
from io import StringIO

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


def strio_to_file(strio: StringIO, file: "str", mode: str = "w") -> None:
    if world.Get_rank() == 0:
        with open(file, mode) as of:
            of.write(strio.getvalue())
    world.Barrier()
