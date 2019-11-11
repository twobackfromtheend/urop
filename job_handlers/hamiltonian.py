import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional

import quimb as q
from scipy.sparse import csr_matrix, save_npz, load_npz

from qubit_system.geometry import BaseGeometry

QType = csr_matrix

HAMILTONIANS_DIR = Path(__file__).parent / "hamiltonians"


@dataclass
class SpinHamiltonian:
    N: int
    time_independent_terms: List[Tuple[int, int, QType]]
    Omega_coeff_terms: QType
    Delta_coeff_terms: QType
    time_independent_terms_sum: Optional[QType] = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: N={self.N}, time_independent_terms: {len(self.time_independent_terms)}" + \
               f"\n\tOmega_coeff_terms: {repr(self.Omega_coeff_terms)}" + \
               f"\n\tDelta_coeff_terms: {repr(self.Delta_coeff_terms)}"

    def save(self):
        save_dir: Path = HAMILTONIANS_DIR / str(self.N)
        save_dir.mkdir()

        for i, j, n_ij in self.time_independent_terms:
            ti_save_file: Path = save_dir / f"ti_{i}_{j}.npz"
            with ti_save_file.open("wb") as f:
                save_npz(f, n_ij)

        Omega_save_file: Path = save_dir / f"Omega.npz"
        with Omega_save_file.open("wb") as f:
            save_npz(f, self.Omega_coeff_terms)

        Delta_save_file: Path = save_dir / f"Delta.npz"
        with Delta_save_file.open("wb") as f:
            save_npz(f, self.Delta_coeff_terms)

    @staticmethod
    def load(N: int) -> 'SpinHamiltonian':
        save_dir: Path = HAMILTONIANS_DIR / str(N)
        if not save_dir.is_dir():
            raise FileNotFoundError(f"Cannot load Hamiltonian for N={N} as folder does not exist.")
        time_independent_terms = []

        for filepath in save_dir.glob("ti_*.npz"):
            match = re.match(r"ti_(\d*)_(\d*).npz", filepath.name)
            if match:
                i = int(match.group(1))
                j = int(match.group(2))
                with filepath.open("rb") as f:
                    n_ij: csr_matrix = load_npz(f)
                time_independent_terms.append((i, j, n_ij))
            else:
                raise ValueError(f"Unknown file: {filepath} when parsing time_independent_terms")

        Omega_save_file: Path = save_dir / f"Omega.npz"
        with Omega_save_file.open("rb") as f:
            Omega_coeff_terms: csr_matrix = load_npz(f)
        Delta_save_file: Path = save_dir / f"Delta.npz"
        with Delta_save_file.open("rb") as f:
            Delta_coeff_terms: csr_matrix = load_npz(f)

        return SpinHamiltonian(
            N=N,
            time_independent_terms=time_independent_terms,
            Omega_coeff_terms=Omega_coeff_terms,
            Delta_coeff_terms=Delta_coeff_terms
        )

    @staticmethod
    def create_and_save_hamiltonian(N: int) -> 'SpinHamiltonian':
        sx = q.pauli("X", sparse=True)
        sz = q.pauli("Z", sparse=True)
        qnum = (sz + q.identity(2, sparse=True)) / 2
        dims = [2] * N

        # noinspection PyTypeChecker
        time_independent_terms: List[Tuple[int, int, QType]] = []
        # noinspection PyTypeChecker
        Omega_coeff_terms: QType = 0
        # noinspection PyTypeChecker
        Delta_coeff_terms: QType = 0

        for i in range(N):
            Omega_coeff_terms += q.ikron(sx, dims=dims, inds=i, sparse=True) / 2
            n_i = q.ikron(qnum, dims=dims, inds=i, sparse=True)
            Delta_coeff_terms -= n_i

            for j in range(i):
                n_j = q.ikron(qnum, dims=dims, inds=j, sparse=True)

                time_independent_terms.append((i, j, n_i * n_j))
        spin_ham = SpinHamiltonian(
            N=N,
            time_independent_terms=time_independent_terms,
            Omega_coeff_terms=Omega_coeff_terms,
            Delta_coeff_terms=Delta_coeff_terms
        )
        spin_ham.save()
        return spin_ham

    def get_hamiltonian(self, V: float, geometry: BaseGeometry, Omega: float, Delta: float) -> QType:
        # noinspection PyTypeChecker
        if self.time_independent_terms_sum is None:
            self.time_independent_terms_sum: QType = 0
            for i, j, n_ij in self.time_independent_terms:
                self.time_independent_terms_sum += V / geometry.get_distance(i, j) ** 6 * n_ij

        return self.time_independent_terms_sum + Omega * self.Omega_coeff_terms + Delta * self.Delta_coeff_terms


if __name__ == '__main__':
    # SpinHamiltonian.create_and_save_hamiltonian(8)
    spin_ham = SpinHamiltonian.load(12)
    print(spin_ham)
