from typing import Tuple, List, Optional, Union

import numpy as np
import quimb as q

from qubit_system.geometry.base_noisy_geometry import BaseNoisyGeometry
from qubit_system.utils import states
from qubit_system.utils.ghz_states import BaseGHZState


class EvolvingQubitSystemWithGeometryNoise:
    def __init__(self, N: int, V: float, geometry: BaseNoisyGeometry, t_list: np.ndarray, ghz_state: BaseGHZState,
                 solve_points_per_timestep: int = 1):
        self.N = N
        self.V = V

        self.geometry = geometry

        self.t_list = t_list
        self.required_control_timesteps = len(t_list) - 1

        self.psi_0 = q.kron(*states.get_ground_states(N))
        self.ghz_state = ghz_state

        self._hamiltonian_variables = self._get_hamiltonian_variables()
        self.iteration_hamiltonian_variables = self._get_iteration_hamiltonian_variables()

        self.solve_points_per_timestep = solve_points_per_timestep

        self.evo: Optional[q.Evolution] = None
        self.solved_states: List[q.qarray] = []
        self.solved_t_list = []

    def _get_iteration_hamiltonian_variables(self):
        """
        Called every iteration to regenerate Hamiltonian based on newly randomised qubit locations.
        :return:
        """
        # noinspection PyTypeChecker
        time_independent_terms: q.qarray = 0
        for i, j, n_ij in self._hamiltonian_variables[0]:
            time_independent_terms += self.V / self.geometry.get_distance(i, j, with_noise=True) ** 6 * n_ij

        return (
            time_independent_terms,
            self._hamiltonian_variables[1],
            self._hamiltonian_variables[2]
        )

    def _get_hamiltonian(self, Omega: float, Delta: float) -> q.qarray:
        """
        Called at every timestep to obtain a time-independent Hamiltonian given the control fields at that time
        :param Omega:
        :param Delta:
        :return:
        """
        return self.iteration_hamiltonian_variables[0] \
               + Omega * self.iteration_hamiltonian_variables[1] \
               + Delta * self.iteration_hamiltonian_variables[2]

    def _get_hamiltonian_variables(self) -> Tuple[List[Tuple[int, int, q.qarray]], q.qarray, q.qarray]:
        sx = q.pauli("X", sparse=True)
        sz = q.pauli("Z", sparse=True)
        qnum = (sz + q.identity(2, sparse=True)) / 2
        dims = [2] * self.N

        # noinspection PyTypeChecker
        time_independent_terms: List[Tuple[int, int, q.qarray]] = []
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

                time_independent_terms.append((i, j, n_i * n_j))

        return (
            time_independent_terms,
            Omega_coeff_terms,
            Delta_coeff_terms
        )

    def solve_with_protocol(self, Omega: np.ndarray, Delta: np.ndarray) -> List[q.qarray]:
        assert self.required_control_timesteps == len(Omega) == len(Delta), \
            "Omega and Delta need to be of equal length, and of length one less than t_list"
        self.geometry.reset_coordinates()
        self.iteration_hamiltonian_variables = self._get_iteration_hamiltonian_variables()

        dt = self.t_list[1]

        self.solved_states = [self.psi_0]
        self.solved_t_list = [0]

        latest_state = state = self.psi_0
        latest_time = 0
        # for i in tqdm(range(len(self.Omega))):
        for i in range(len(Omega)):
            _Omega = Omega[i]
            _Delta = Delta[i]
            self.evo = q.Evolution(
                latest_state,
                self._get_hamiltonian(_Omega, _Delta),
                method="integrate",
            )
            solve_points = np.linspace(0, dt, self.solve_points_per_timestep + 1)[1:]  # Take away t=0 as a solve point
            for state in self.evo.at_times(solve_points):
                self.solved_states.append(state)
            self.solved_t_list += (solve_points + latest_time).tolist()
            latest_state = state
            latest_time = self.solved_t_list[-1]

        return self.solved_states

    def get_fidelity_with(self, target_state: Union[str, q.qarray] = "ghz") -> float:
        """
        :param target_state:
            One of "ghz", "ghz_antisymmetric", "ground", and "excited".
            Can also be ghz_component_1 or ghz_component_2
        :return:
        """
        assert (self.evo is not None), "evo attribute cannot be None (call solve method)"

        final_state = self.solved_states[-1]
        if target_state == "ghz":
            return q.fidelity(final_state, self.ghz_state.get_state_tensor(symmetric=True))
        elif target_state == "ghz_antisymmetric":
            return q.fidelity(final_state, self.ghz_state.get_state_tensor(symmetric=False))
        elif target_state == "ghz_component_1":
            return q.fidelity(final_state, self.ghz_state._get_components()[0])
        elif target_state == "ghz_component_2":
            return q.fidelity(final_state, self.ghz_state._get_components()[1])
        elif target_state == "ground":
            return q.fidelity(final_state, q.kron(*states.get_ground_states(self.N)))
        elif target_state == "excited":
            return q.fidelity(final_state, q.kron(*states.get_excited_states(self.N)))
        elif isinstance(target_state, q.qarray):
            return q.fidelity(final_state, target_state)
        else:
            raise ValueError(f"target_state has to be one of 'ghz', 'ground', or 'excited', not {target_state}.")
