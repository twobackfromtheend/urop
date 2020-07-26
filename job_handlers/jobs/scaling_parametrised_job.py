import gc
import os
import time
from functools import partial

import numpy as np
import quimb as q
from numba import jit

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
    'BO_MAX_ITER': '300',
    'BO_EXPLOIT_ITER': '10',
    'SETUP': '1d_std',
    'TUKEY_ALPHA': '0.8',
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
max_iter = int(getenv("BO_MAX_ITER"))
exploit_iter = int(getenv("BO_EXPLOIT_ITER"))

tukey_alpha = eval(getenv("TUKEY_ALPHA"))
setup = getenv("SETUP")
do_only_16 = os.getenv(eval("ONLY_16"), False)

print(
    "Parameters:\n"
    f"\tjob_id: {job_id}\n"
    f"\tTUKEY_ALPHA: {tukey_alpha}\n"
    f"\tBO_MAX_ITER: {max_iter}\n"
    f"\tBO_EXPLOIT_ITER: {exploit_iter}\n"
    f"\tSETUP: {setup}\n"
)


@jit(nopython=True)
def _get_figure_of_merit(state, state_index_1: int, state_index_2: int):
    density_matrix = state @ state.conjugate().transpose()
    rho_00 = density_matrix[state_index_1, state_index_1].real
    rho_11 = density_matrix[state_index_2, state_index_2].real
    off_diag_1 = density_matrix[state_index_2, state_index_1]
    off_diagonal = 2 * np.abs(off_diag_1).real
    result = (rho_00 + rho_11 + off_diagonal) / 2
    return result, rho_00, rho_11, off_diagonal

def get_f(spin_ham: SpinHamiltonian, V: float, geometry: BaseGeometry,
          t_list: np.ndarray, psi_0: QType,
          ghz_state: BaseGHZState,
          protocol_generator: BaseProtocolGenerator):
    # ghz_state_tensor = ghz_state.get_state_tensor()

    ghz_components = ghz_state._get_components()
    state_index_1 = ghz_components[0].argmax()
    state_index_2 = ghz_components[1].argmax()

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
            ghz_fidelity, rho_00, rho_11, off_diag = _get_figure_of_merit(final_state, state_index_1, state_index_2)
            # ghz_fidelity = q.fidelity(final_state, ghz_state_tensor)
            print(f"fidelity: {ghz_fidelity:.3f} for input: {input_}")
            return 1 - ghz_fidelity

        output = np.array([get_figure_of_merit(input_) for input_ in inputs])
        print(f"func f completed in {time.time() - start_time:.3f}s, output: {output}")
        gc.collect()
        return output

    return f


setup_details = {
    '1d_std': {
        4: (
            RegularLattice((4, 1), spacing=LATTICE_SPACING),
            CustomGHZState(4, [True] * 4)
        ),
        8: (
            RegularLattice((8, 1), spacing=LATTICE_SPACING),
            CustomGHZState(8, [True] * 8)
        ),
        12: (
            RegularLattice((12, 1), spacing=LATTICE_SPACING),
            CustomGHZState(12, [True] * 12)
        ),
        16: (
            RegularLattice((16, 1), spacing=LATTICE_SPACING),
            CustomGHZState(16, [True] * 16)
        ),
    },
    '2d_std': {
        4: (
            RegularLattice((2, 2), spacing=LATTICE_SPACING),
            CustomGHZState(4, [True] * 4)
        ),
        8: (
            RegularLattice((4, 2), spacing=LATTICE_SPACING),
            CustomGHZState(8, [True] * 8)
        ),
        12: (
            RegularLattice((4, 3), spacing=LATTICE_SPACING),
            CustomGHZState(12, [True] * 12)
        ),
        16: (
            RegularLattice((4, 4), spacing=LATTICE_SPACING),
            CustomGHZState(16, [True] * 16)
        ),
    },
    '3d_std': {
        8: (
            RegularLattice((2, 2, 2), spacing=LATTICE_SPACING),
            CustomGHZState(8, [True] * 8)
        ),
        12: (
            RegularLattice((3, 2, 2), spacing=LATTICE_SPACING),
            CustomGHZState(12, [True] * 12)
        ),
        16: (
            RegularLattice((4, 2, 2), spacing=LATTICE_SPACING),
            CustomGHZState(16, [True] * 16)
        ),
    },
    '1d_alt': {
        4: (
            RegularLattice((4, 1), spacing=LATTICE_SPACING),
            CustomGHZState(4, [True, False] * 2)
        ),
        8: (
            RegularLattice((8, 1), spacing=LATTICE_SPACING),
            CustomGHZState(8, [True, False] * 4)
        ),
        12: (
            RegularLattice((12, 1), spacing=LATTICE_SPACING),
            CustomGHZState(12, [True, False] * 6)
        ),
        16: (
            RegularLattice((16, 1), spacing=LATTICE_SPACING),
            CustomGHZState(16, [True, False] * 8)
        ),
    },
    '2d_alt': {
        4: (
            RegularLattice((2, 2), spacing=LATTICE_SPACING),
            CustomGHZState(4, [True, False, False, True])
        ),
        8: (
            RegularLattice((4, 2), spacing=LATTICE_SPACING),
            CustomGHZState(8, [True, False, False, True] * 2)
        ),
        12: (
            RegularLattice((4, 3), spacing=LATTICE_SPACING),
            CustomGHZState(12, [True, False, True, False, True, False] * 2)
        ),
        16: (
            RegularLattice((4, 4), spacing=LATTICE_SPACING),
            CustomGHZState(16, [True, False, True, False, False, True, False, True] * 2)
        ),
    },
    '3d_alt': {
        8: (
            RegularLattice((2, 2, 2), spacing=LATTICE_SPACING),
            CustomGHZState(8, [True, False, False, True, False, True, True, False])
        ),
        12: (
            RegularLattice((3, 2, 2), spacing=LATTICE_SPACING),
            CustomGHZState(12, [True, False, False, True, False, True, True, False, True, False, False, True])
        ),
        16: (
            RegularLattice((4, 2, 2), spacing=LATTICE_SPACING),
            CustomGHZState(16, [True, False, False, True, False, True, True, False] * 2)
        ),
    },
}


if __name__ == '__main__':
    np.set_printoptions(linewidth=200, precision=None, floatmode='maxprec')

    trigger_event("job_progress", value1="Job started", value2=job_id)

    protocol_timesteps = 3
    t = 1e-6 if 'alt' in setup else 1e-7
    interpolation_timesteps = 3000
    t_list = np.linspace(0, t, interpolation_timesteps + 1)

    for N, (geometry, ghz_state) in setup_details[setup].items():
        if do_only_16:
            if N != 16:
                continue
        with timer(f"Loading SpinHam (N={N})"):
            spin_ham = SpinHamiltonian.load(N)

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
        print(f"END FOR N={N}")

    trigger_event("job_progress", value1="Job ended", value2=job_id)
