import os
import time
from functools import partial

import numpy as np
import quimb as q

import interaction_constants
from ifttt_webhook import trigger_event
from job_handlers.crossing import get_ghz_crossing
from job_handlers.hamiltonian import SpinHamiltonian, QType
from job_handlers.jobs.bo_utils import get_domain, optimise
from job_handlers.solver import solve_with_protocol
from job_handlers.timer import timer
from protocol_generator.noisy_interpolation_pg import NoisyInterpolationPG
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
    'Q_GEOMETRY': 'NoisyRegularLattice(shape=(4, 2), spacing=LATTICE_SPACING, spacing_noise=0.1e-6)',
    'Q_GHZ_STATE': 'CustomGHZState(N, [True, False, False, True, True, False, False, True])',
    # 'N': '8',
    # 'Q_GEOMETRY': 'RegularLattice(shape=(4, 2), spacing=LATTICE_SPACING)',
    # 'Q_GHZ_STATE': 'CustomGHZState(N, [True, False, False, True, True, False, False, True])',
    # 'N': '20',
    # 'Q_GEOMETRY': 'RegularLattice2D(shape=(4, 5), spacing=LATTICE_SPACING)',
    # 'Q_GHZ_STATE': 'CustomGHZState(N, [True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False])',

    'Q_F_NOISE': 0.01,
    'Q_PROTOCOL_NOISE': 0.01,

    'ITERS_PER_PARAM': 1,

    'BO_MAX_ITER': '50',
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

f_noise = float(getenv("Q_F_NOISE"))
protocol_noise = float(getenv("Q_PROTOCOL_NOISE"))

iters_per_param = int(getenv("ITERS_PER_PARAM"))

max_iter = int(getenv("BO_MAX_ITER"))
exploit_iter = int(getenv("BO_EXPLOIT_ITER"))

print(
    "Parameters:\n"
    f"\tjob_id: {job_id}\n"
    f"\tN: {N}\n"
    f"\tQ_GHZ_STATE: {ghz_state} ({ghz_state_envvar})\n"
    f"\tQ_GEOMETRY: {geometry} ({geometry_envvar})\n"

    f"\tQ_F_NOISE: {f_noise}\n"
    f"\tQ_PROTOCOL_NOISE: {protocol_noise}\n"

    f"\tITERS_PER_PARAM: {iters_per_param}\n"
    f"\tBO_MAX_ITER: {max_iter}\n"
    f"\tBO_EXPLOIT_ITER: {exploit_iter}\n"
)

assert isinstance(geometry, BaseNoisyGeometry), "Geometry must be subclass of BaseNoisyGeometry"


def get_f(spin_ham: SpinHamiltonian, V: float, geometry: BaseNoisyGeometry,
          t_list: np.ndarray, psi_0: QType,
          ghz_state: BaseGHZState,
          protocol_generator: NoisyInterpolationPG):
    ghz_state_tensor = ghz_state.get_state_tensor()

    def f(inputs: np.ndarray) -> np.ndarray:
        """
        :param inputs: 2-dimensional array
        :return: 2-dimentional array, one-evaluation per row
        """
        print(f"inputs: {inputs}")
        start_time = time.time()

        def get_figure_of_merit(input_: np.ndarray):
            ghz_fidelities = []
            for i in range(iters_per_param):
                # Reset geometry and generate new Hamiltonian (time-independent terms need to be resummed)
                geometry.reset_coordinates()
                spin_ham.reset_geometry()

                Omega, Delta = protocol_generator.get_protocol(input_)
                final_state = solve_with_protocol(
                    spin_ham, V=V, geometry=geometry, t_list=t_list, psi_0=psi_0,
                    Omega=Omega, Delta=Delta
                )[-1]
                ghz_fidelity = q.fidelity(final_state, ghz_state_tensor)
                print(f"fidelity: {ghz_fidelity:.3f} for input: {input_}")
                ghz_fidelities.append(ghz_fidelity)

            ghz_fidelities = np.array(ghz_fidelities)
            noisy_ghz_fidelities = ghz_fidelities + np.random.normal(
                size=ghz_fidelities.shape) * ghz_fidelities * f_noise
            mean_ghz_fidelity = np.mean(noisy_ghz_fidelities)
            return 1 - mean_ghz_fidelity

        output = np.array([get_figure_of_merit(input_) for input_ in inputs])
        print(f"func f completed in {time.time() - start_time:.3f}s, output: {output}")
        return output

    return f


if __name__ == '__main__':
    trigger_event("job_progress", value1="Job started", value2=job_id)

    protocol_timesteps = 3
    t = 2e-6
    interpolation_timesteps = 3000
    t_list = np.linspace(0, t, interpolation_timesteps + 1)

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

    protocol_generator = NoisyInterpolationPG(t_list, kind="quadratic", noise=protocol_noise)

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
