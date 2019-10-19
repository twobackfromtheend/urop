from typing import Tuple

import quimb as q

from qubit_system.geometry import BaseGeometry


class BaseQubitSystem:

    def __init__(self, N: int, V: float, geometry: BaseGeometry):
        self.N = N
        self.V = V

        self.geometry = geometry
        self._hamiltonian_variables = None

    def _get_hamiltonian(self, Omega: float, Delta: float) -> q.qarray:
        if self._hamiltonian_variables is None:
            self._hamiltonian_variables = self._get_hamiltonian_variables()

        return self._hamiltonian_variables[0] \
               + Omega * self._hamiltonian_variables[1] \
               + Delta * self._hamiltonian_variables[2]

    def _get_hamiltonian_variables(self) -> Tuple[q.qarray, q.qarray, q.qarray]:
        sx = q.pauli("X", sparse=True)
        sz = q.pauli("Z", sparse=True)
        qnum = (sz + q.identity(2, sparse=True)) / 2
        dims = [2] * self.N

        # noinspection PyTypeChecker
        time_independent_terms: q.qarray = 0
        # noinspection PyTypeChecker
        Omega_coeff_terms: q.qarray = 0
        # noinspection PyTypeChecker
        Delta_coeff_terms: q.qarray = 0

        for i in range(self.N):
            Omega_coeff_terms += q.ikron(sx, dims=dims, inds=i, sparse=True) / 2
            n_i = q.ikron(qnum, dims=dims, inds=i, sparse=True)
            Delta_coeff_terms -= n_i

            for j in range(i):
                n_j = q.ikron(qnum, dims=dims, inds=j, sparse=True)

                time_independent_terms += self.V / self.geometry.get_distance(i, j) ** 6 * n_i * n_j

        return (
            time_independent_terms,
            Omega_coeff_terms,
            Delta_coeff_terms
        )