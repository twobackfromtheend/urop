import gc
import os
import time
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple, Optional

import numpy as np
import quimb as q

import interaction_constants
from ifttt_webhook import trigger_event
from job_handlers.crossing import get_ghz_crossing
from job_handlers.hamiltonian import SpinHamiltonian, QType
from job_handlers.jobs.bo_utils import get_domain, optimise
from job_handlers.solver import solve_with_protocol
from job_handlers.timer import timer
from protocol_generator.base_protocol_generator import BaseProtocolGenerator
from protocol_generator.interpolation_pg import InterpolationPG
from qubit_system.geometry import *
from qubit_system.utils import states
from qubit_system.utils.ghz_states import *

# Flush prints immediately.
print = partial(print, flush=True)

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

LATTICE_SPACING = 1.5e-6

print(f"C6: {C6:.3e}")
characteristic_V = C6 / (LATTICE_SPACING ** 6)
print(f"Characteristic V: {characteristic_V:.3e} Hz")

LOCAL_JOB_ENVVARS = {
    'PBS_JOBID': 'LOCAL_JOB',
    'N': '8',

    # 'Q_GEOMETRY': 'DoubleRing(8, spacing=LATTICE_SPACING)',
    # 'Q_GHZ_STATE': 'CustomGHZState(N, [True, False, True, False, False, True, False, True])',

    # 'Q_GEOMETRY': 'Star(8, spacing=LATTICE_SPACING)',
    # 'Q_GHZ_STATE': 'CustomGHZState(N, [True, False, True, False, False, True, False, True])',

    'Q_GEOMETRY': 'RegularLattice(shape=(8,), spacing=LATTICE_SPACING)',
    # 'Q_GHZ_STATE': 'CustomGHZState(N, [True, True, True, True, True, True, True, True])',
    'Q_GHZ_STATE': 'CustomGHZState(N, [True, False, True, False, True, False, True, False])',

    # 'Q_GEOMETRY': 'RegularLattice(shape=(4, 2), spacing=LATTICE_SPACING)',
    # 'Q_GHZ_STATE': 'CustomGHZState(N, [True, False, False, True, True, False, False, True])',
    # 'N': '8',
    # 'Q_GEOMETRY': 'RegularLattice(shape=(4, 2), spacing=LATTICE_SPACING)',
    # 'Q_GHZ_STATE': 'CustomGHZState(N, [True, False, False, True, True, False, False, True])',
    # 'N': '20',
    # 'Q_GEOMETRY': 'RegularLattice2D(shape=(4, 5), spacing=LATTICE_SPACING)',
    # 'Q_GHZ_STATE': 'CustomGHZState(N, [True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False])',
    'BO_MAX_ITER': '300',
    'BO_EXPLOIT_ITER': '10',
}

IS_LOCAL_JOB = not bool(os.getenv("PBS_JOBID"))
print(f"IS_LOCAL_JOB: {IS_LOCAL_JOB}")


def getenv(key: str):
    if not IS_LOCAL_JOB:
        os_env = os.getenv(key)
        if os_env is not None:
            return os_env
        else:
            print(f"Could not get envvar {key}, using {LOCAL_JOB_ENVVARS[key]}")
            return LOCAL_JOB_ENVVARS[key]
    else:
        return LOCAL_JOB_ENVVARS[key]


job_id = getenv("PBS_JOBID")
N = int(getenv("N"))
geometry_envvar = getenv("Q_GEOMETRY")
geometry = eval(geometry_envvar)
ghz_state_envvar = getenv("Q_GHZ_STATE")
ghz_state = eval(ghz_state_envvar)

max_iter = int(getenv("BO_MAX_ITER"))
exploit_iter = int(getenv("BO_EXPLOIT_ITER"))

print(
    "Parameters:\n"
    f"\tjob_id: {job_id}\n"
    f"\tN: {N}\n"
    f"\tQ_GEOMETRY: {geometry} ({geometry_envvar})\n"
    f"\tQ_GHZ_STATE: {ghz_state} ({ghz_state_envvar})\n"
    f"\tBO_MAX_ITER: {max_iter}\n"
    f"\tBO_EXPLOIT_ITER: {exploit_iter}\n"
)


def get_f(spin_ham: SpinHamiltonian, V: float, geometry: BaseGeometry,
          t_list: np.ndarray, psi_0: QType,
          ghz_state: BaseGHZState,
          protocol_generator: BaseProtocolGenerator):
    ghz_state_tensor = ghz_state.get_state_tensor()

    def f(inputs: np.ndarray) -> np.ndarray:
        """
        :param inputs: 2-dimensional array
        :return: 2-dimentional array, one-evaluation per row
        """
        print(f"inputs: {inputs}")
        start_time = time.time()

        def get_figure_of_merit(input_: np.ndarray):
            Omega, Delta = protocol_generator.get_protocol(input_)
            final_state = solve_with_protocol(
                spin_ham, V=V, geometry=geometry, t_list=t_list, psi_0=psi_0,
                Omega=Omega, Delta=Delta
            )
            ghz_fidelity = q.fidelity(final_state, ghz_state_tensor)
            print(f"fidelity: {ghz_fidelity:.3f} for input: {input_}")
            return 1 - ghz_fidelity

        output = np.array([get_figure_of_merit(input_) for input_ in inputs])
        print(f"func f completed in {time.time() - start_time:.3f}s, output: {output}")
        gc.collect()
        return output

    return f


@dataclass
class SpinHamiltonianWithFields(SpinHamiltonian):
    offset_field_terms: QType = None

    @staticmethod
    def load(N: int) -> 'SpinHamiltonianWithFields':
        sx = q.pauli("X", sparse=True)
        sz = q.pauli("Z", sparse=True)
        qnum = (sz + q.identity(2, sparse=True)) / 2
        dims = [2] * N

        time_independent_terms: List[Tuple[int, int, QType]] = []
        Omega_coeff_terms: QType = 0
        Delta_coeff_terms: QType = 0
        offset_field_terms: QType = 0

        for i in range(N):
            Omega_coeff_terms += q.ikron(sx, dims=dims, inds=i, sparse=True) / 2
            n_i = q.ikron(qnum, dims=dims, inds=i, sparse=True)
            Delta_coeff_terms -= n_i
            offset_field_terms -= SpinHamiltonianWithFields._get_offset_field(N, i) * n_i

            for j in range(i):
                n_j = q.ikron(qnum, dims=dims, inds=j, sparse=True)

                time_independent_terms.append((i, j, n_i * n_j))
        return SpinHamiltonianWithFields(
            N=N,
            time_independent_terms=time_independent_terms,
            Omega_coeff_terms=Omega_coeff_terms,
            Delta_coeff_terms=Delta_coeff_terms,
            offset_field_terms=offset_field_terms,
        )

    @staticmethod
    def _get_offset_field(N: int, i: int) -> float:
        if N <= 8:
            if i == 0 or i == N - 1:
                return -4.5e6 * 2 * np.pi
        else:
            if i == 0 or i == N - 1:
                return -6e6 * 2 * np.pi
            if i == 3 or i == N - 4:
                return -1.5e6 * 2 * np.pi
        return 0

    def get_hamiltonian(self, V: float, geometry: BaseGeometry, Omega: float, Delta: float) -> QType:
        return super().get_hamiltonian(V, geometry, Omega, Delta) + self.offset_field_terms

    @staticmethod
    def create_and_save_hamiltonian(N: int) -> 'SpinHamiltonian':
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def reset_geometry(self):
        raise NotImplementedError


if __name__ == '__main__':
    np.set_printoptions(linewidth=200, precision=None, floatmode='maxprec')

    trigger_event("job_progress", value1="Job started", value2=job_id)

    protocol_timesteps = 3
    t = 1e-6
    interpolation_timesteps = 3000
    t_list = np.linspace(0, t, interpolation_timesteps + 1)

    with timer(f"Loading SpinHam (N={N})"):
        spin_ham = SpinHamiltonianWithFields.load(N)

    with timer(f"Calculating crossing"):
        crossing = get_ghz_crossing(
            spin_ham=spin_ham, characteristic_V=characteristic_V,
            ghz_state=ghz_state, geometry=geometry,
            V=C6
        )
    Omega_limits = (0, crossing)
    Delta_limits = (0.5 * crossing, 1.5 * crossing)
    domain = get_domain(Omega_limits, Delta_limits, protocol_timesteps)

    protocol_generator = InterpolationPG(t_list, kind="cubic")

    with timer(f"Getting f"):
        f = get_f(
            spin_ham=spin_ham,
            V=C6,
            geometry=geometry,
            t_list=t_list,
            psi_0=q.kron(*states.get_ground_states(N)),
            ghz_state=ghz_state,
            protocol_generator=protocol_generator,
        )

    with timer(f"Optimising f"):
        bo = optimise(f, domain, max_iter=max_iter, exploit_iter=exploit_iter)

    print("x_opt", bo.x_opt)
    print("fx_opt", bo.fx_opt)

    trigger_event("job_progress", value1="Job ended", value2=job_id)
