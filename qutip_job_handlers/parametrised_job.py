import os
import time
from functools import partial
from typing import Callable

import numpy as np
from qutip import Qobj, fidelity, tensor, basis
from scipy.interpolate import interp1d

import interaction_constants
from qubit_system.geometry import *
from qutip_job_handlers.hamiltonian import QutipSpinHamiltonian
from qutip_job_handlers.qutip_states import *
from qutip_job_handlers.solver import solve
from windows.tukey_window import tukey

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
    'Q_GEOMETRY': 'RegularLattice(shape=(4, 2), spacing=LATTICE_SPACING)',
    'Q_GHZ_STATE': 'CustomGHZState(N, [True, False, False, True, True, False, False, True])',
    # 'N': '8',
    # 'Q_GEOMETRY': 'RegularLattice(shape=(4, 2), spacing=LATTICE_SPACING)',
    # 'Q_GHZ_STATE': 'CustomGHZState(N, [True, False, False, True, True, False, False, True])',
    # 'N': '20',
    # 'Q_GEOMETRY': 'RegularLattice2D(shape=(4, 5), spacing=LATTICE_SPACING)',
    # 'Q_GHZ_STATE': 'CustomGHZState(N, [True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False])',
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


def get_f(spin_ham: QutipSpinHamiltonian, V: float, geometry: BaseGeometry,
          t_list: np.ndarray, psi_0: Qobj,
          ghz_state: BaseGHZState,
          ):
    ghz_state_tensor = ghz_state.get_state_tensor()
    timesteps = len(t_list)
    t = t_list[-1]

    def f(inputs: np.ndarray) -> np.ndarray:
        """
        :param inputs: 2-dimensional array
        :return: 2-dimentional array, one-evaluation per row
        """
        print(f"inputs: {inputs}")
        start_time = time.time()

        def get_figure_of_merit(input_: np.ndarray):
            num_params = len(input_) // 2
            Omega_params = input_[:num_params]
            Delta_params = input_[num_params:]
            input_t_list = np.linspace(0, t_list[-1], num_params + 1)

            Omega_func: Callable[[float], float] = interp1d(input_t_list, np.hstack((Omega_params, Omega_params[-1])),
                                                            kind="quadratic", bounds_error=False, fill_value=0)
            timesteps = len(t_list)
            window_fn = tukey(timesteps, alpha=0.2)

            def Omega(x: float, *args) -> float:
                return Omega_func(x) * window_fn(timesteps * x / t)

            Delta_func: Callable[[float], float] = interp1d(input_t_list, np.hstack((Delta_params, Delta_params[-1])),
                                                            kind="quadratic", bounds_error=False, fill_value=0)

            def Delta(x: float, *args) -> float:
                return Delta_func(x)

            hamiltonian = spin_ham.get_hamiltonian(V, geometry, Omega, Delta)

            final_state = solve(hamiltonian, psi_0, t_list).states[-1]
            ghz_fidelity = fidelity(final_state, ghz_state_tensor) ** 2
            print(f"fidelity: {ghz_fidelity:.3f} for input: {input_}")
            return 1 - ghz_fidelity

        output = np.array([get_figure_of_merit(input_) for input_ in inputs])
        print(f"func f completed in {time.time() - start_time:.3f}s, output: {output}")
        return output

    return f


if __name__ == '__main__':
    from ifttt_webhook import trigger_event
    from qutip_job_handlers.crossing import get_ghz_crossing
    from job_handlers.jobs.bo_utils import get_domain, optimise
    from job_handlers.timer import timer

    trigger_event("job_progress", value1="Job started", value2=job_id)

    protocol_timesteps = 3
    t = 2e-6
    interpolation_timesteps = 3000
    t_list = np.linspace(0, t, interpolation_timesteps + 1)

    with timer(f"Creating QutipSpinHam (N={N})"):
        spin_ham = QutipSpinHamiltonian(N)

    with timer(f"Calculating crossing"):
        crossing = get_ghz_crossing(
            spin_ham=spin_ham, characteristic_V=characteristic_V,
            ghz_state=ghz_state, geometry=geometry,
            V=C6
        )
    Omega_limits = (0, crossing)
    Delta_limits = (0.5 * crossing, 1.5 * crossing)
    domain = get_domain(Omega_limits, Delta_limits, protocol_timesteps)

    psi_0 = tensor([basis(2, 1) for _ in range(N)])

    with timer(f"Getting f"):
        f = get_f(
            spin_ham=spin_ham,
            V=C6,
            geometry=geometry,
            t_list=t_list,
            psi_0=psi_0,
            ghz_state=ghz_state,
        )

    with timer(f"Optimising f"):
        bo = optimise(f, domain, max_iter=max_iter, exploit_iter=exploit_iter)

    print("x_opt", bo.x_opt)
    print("fx_opt", bo.fx_opt)

    trigger_event("job_progress", value1="Job ended", value2=job_id)
