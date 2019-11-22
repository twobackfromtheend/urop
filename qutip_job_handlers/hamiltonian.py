from dataclasses import dataclass
from typing import Callable, List, Tuple

from qutip import *

from qubit_system.geometry import BaseGeometry
from qutip_job_handlers.qutip_utils import get_exp_list


@dataclass
class QutipSpinHamiltonian:
    def __init__(self, N: int):
        self.N = N

        sx_list, sy_list, sz_list = get_exp_list(N)

        self.time_independent_terms: List[Tuple[int, int, Qobj]] = []
        # noinspection PyTypeChecker
        self.Omega_coeff_terms: Qobj = 0
        # noinspection PyTypeChecker
        self.Delta_coeff_terms: Qobj = 0

        for i in range(N):
            self.Omega_coeff_terms += 1 / 2 * sx_list[i]
            n_i = (sz_list[i] + qeye(1)) / 2
            self.Delta_coeff_terms -= n_i

            for j in range(i):
                n_j = (sz_list[j] + qeye(1)) / 2

                self.time_independent_terms.append((i, j, n_i * n_j))

    def get_hamiltonian(self, V: float, geometry: BaseGeometry, Omega: Callable, Delta: Callable):
        time_independent_terms_sum = 0
        for i, j, n_ij in self.time_independent_terms:
            time_independent_terms_sum += V / geometry.get_distance(i, j) ** 6 * n_ij

        return [
            time_independent_terms_sum,
            [self.Omega_coeff_terms, Omega],
            [self.Delta_coeff_terms, Delta]
        ]
