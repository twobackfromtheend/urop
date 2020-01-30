import os
import time
from collections import defaultdict
from typing import Callable

import numpy as np
from qutip import Qobj, fidelity, tensor, basis
from scipy.interpolate import interp1d

import interaction_constants
from qubit_system.geometry import *
from qutip_job_handlers import qutip_utils
from qutip_job_handlers.hamiltonian import QutipSpinHamiltonian
from qutip_job_handlers.qutip_states import *
from qutip_job_handlers.solver import solve
from windows.tukey_window import tukey

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)
LATTICE_SPACING = 1.5e-6

print(f"C6: {C6:.3e}")
characteristic_V = C6 / (LATTICE_SPACING ** 6)
print(f"Characteristic V: {characteristic_V:.3e} Hz")

# Density of the gas given in SI units (atoms/m^3)
# rho = 1.0e12 / (1.0e-6)

# Density of the gas in 3D given in SI units (atoms/m^3)
rho = 1.0e13 / (1.0e-6)
# 1D: rho**(1.0/3.0)    #Given in SI units (atoms/m)
# 2D: rho**(2.0/3.0)    #Given in SI units (atoms/m^2)
rho_new = rho ** (1.0 / 3.0)

# size of the gas in SI units (micrometers)
# 3D: L = (3.0/(4.0*np.pi)*N/rho)**(1.0/3.0)*1.0e6
# 2D: L = (1.0/(np.pi)*N/rho)**(1.0/2.0)*1.0e6
# 1D: L = N/rho*1.0e6


# LOCAL_N = 8  # Number of atoms
LOCAL_N = 9  # Number of atoms

L = LOCAL_N / rho_new * 1.0e6
print(L)

LOCAL_JOB_ENVVARS = {
    'PBS_JOBID': 'LOCAL_JOB',
    'N': LOCAL_N,

    # 'Q_GEOMETRY': 'RegularLattice(shape=(10,), spacing=LATTICE_SPACING)',
    # 'Q_GHZ_STATE': 'CustomGHZState(N, [True, True, True, True, True, True, True, True, True, True])',

    'Q_GEOMETRY': 'RegularLattice(shape=(9,), spacing=LATTICE_SPACING)',
    # 'Q_GEOMETRY': 'RegularLattice(shape=(3, 3), spacing=LATTICE_SPACING)',
    'Q_GHZ_STATE': 'CustomGHZState(N, [True, True, True, True, True, True, True, True, True])',

    # 'Q_GEOMETRY': 'RegularLattice(shape=([8]), spacing=LATTICE_SPACING)',
    # 'Q_GEOMETRY': 'RegularLattice(shape=(4, 2), spacing=LATTICE_SPACING)',
    # 'Q_GEOMETRY': 'RegularLattice(shape=(2, 2, 2), spacing=LATTICE_SPACING)',
    # 'Q_GHZ_STATE': 'CustomGHZState(N, [True, True, True, True, True, True, True, True])',
    # 'Q_GHZ_STATE': 'CustomGHZState(N, [True, False, True, False, True, False, True, False, True])',

    # 'Q_GEOMETRY': 'RegularLattice(shape=(3, 3), spacing=LATTICE_SPACING)',
    # 'Q_GHZ_STATE': 'CustomGHZState(N, [True, False, True, False, True, False, True, False, True])',

    # 'Q_GEOMETRY': 'Random(dimensions=1, rho=rho_new, N=LOCAL_N)',
    # 'Q_GHZ_STATE': 'CustomGHZState(N, [True, False, True, False, True, False, True, False, True])',

    # 'Q_GHZ_STATE': 'CustomGHZState(N, [True, False, True])',

    'BO_MAX_ITER': '50',
    'BO_EXPLOIT_ITER': '0',
    'NUMBER_OF_EXCITED_WANTED': '5',
    'BO_REPEATS_PER_ITER': '5',
    'initial_design_numdata_factor': '4',
}


IS_LOCAL_JOB = not bool(os.getenv("PBS_JOBID"))
print(f"IS_LOCAL_JOB: {IS_LOCAL_JOB}")



def getenv(key: str):
    if not IS_LOCAL_JOB:
        return os.getenv(key)
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
REPEATS_PER_ITER = int(getenv("BO_REPEATS_PER_ITER"))
NUMBER_OF_EXCITED_WANTED = int(getenv("NUMBER_OF_EXCITED_WANTED"))
initial_design_numdata_factor = int(getenv("initial_design_numdata_factor"))

print(
    "Parameters:\n"
    f"\tjob_id: {job_id}\n"
    f"\tN: {N}\n"
    f"\tQ_GEOMETRY: {geometry} ({geometry_envvar})\n"
    f"\tQ_GHZ_STATE: {ghz_state} ({ghz_state_envvar})\n"
    f"\tBO_MAX_ITER: {max_iter}\n"
    f"\tBO_EXPLOIT_ITER: {exploit_iter}\n"
    f"\tBO_REPEATS_PER_ITER: {REPEATS_PER_ITER}\n"
    f"\tNUMBER_OF_EXCITED_WANTED: {NUMBER_OF_EXCITED_WANTED}\n"
    f"\tinitial_design_numdata_factor: {initial_design_numdata_factor}\n"
)


def get_protocol_from_input(input_: np.ndarray, t_list: np.ndarray):
    timesteps = len(t_list)
    t = t_list[-1]

    num_params = len(input_) // 2
    Omega_params = input_[:num_params].copy()
    Delta_params = input_[num_params:].copy()

    protocol_noise = 0.05
    Omega_params += np.random.normal(size=Omega_params.shape) * protocol_noise * Omega_params
    Delta_params += np.random.normal(size=Delta_params.shape) * protocol_noise * Delta_params

    input_t_list = np.linspace(0, t_list[-1], num_params + 1)

    Omega_func: Callable[[float], float] = interp1d(input_t_list, np.hstack((Omega_params, Omega_params[-1])),
                                                    kind="quadratic", bounds_error=False, fill_value=0)
    window_fn = tukey(timesteps, alpha=0.2)

    def Omega(x: float, *args) -> float:
        return Omega_func(x) * window_fn(timesteps * x / t)

    Delta_func: Callable[[float], float] = interp1d(input_t_list, np.hstack((Delta_params, Delta_params[-1])),
                                                    kind="quadratic", bounds_error=False, fill_value=0)

    def Delta(x: float, *args) -> float:
        return Delta_func(x)

    return Omega, Delta


def get_f(spin_ham: QutipSpinHamiltonian, V: float, geometry: BaseGeometry,
          t_list: np.ndarray, psi_0: Qobj,
          ghz_state: BaseGHZState,
          ):
    # ghz_state_tensor = ghz_state.get_state_tensor()

    # To get A_N, or any single component of a CustomGHZState
    # first_ghz_component = ghz_state._get_components()[0]

    def f(inputs: np.ndarray) -> np.ndarray:
        """
        :param inputs: 2-dimensional array
        :return: 2-dimentional array, one-evaluation per row
        """
        print(f"inputs: {inputs}")
        start_time = time.time()

        states_list = qutip_utils.get_states(N)
        state_tensors_by_excited_count = defaultdict(list)
        for state in states_list:
            state_label = qutip_utils.get_label_from_state(state)
            state_excited_count = sum(letter == "e" for letter in state_label)
            state_tensors_by_excited_count[state_excited_count].append(tensor(*state))

        figure_of_merit_kets = state_tensors_by_excited_count[NUMBER_OF_EXCITED_WANTED]

        def get_figure_of_merit(input_: np.ndarray):
            figure_of_merit_over_repeats = []
            for i in range(REPEATS_PER_ITER):
                Omega, Delta = get_protocol_from_input(input_, t_list)

                hamiltonian = spin_ham.get_hamiltonian(V, geometry, Omega, Delta, filling_fraction=0.9)
                solve_result = solve(hamiltonian, psi_0, t_list)
                final_state = solve_result.final_state

                # GHZ fidelity FOM
                # ghz_fidelity = fidelity(final_state, ghz_state_tensor) ** 2
                # print(f"fidelity: {ghz_fidelity:.3f} for input: {input_}")
                # return 1 - ghz_fidelity

                # GHZ_1 FOM (e.g. A_N)
                # ghz_1_fidelity = q.fidelity(final_state, first_ghz_component,squared=True)
                # print(f"fidelity: {ghz_1_fidelity:.3f} for input: {input_}")
                # return 1 - ghz_1_fidelity

                # EXCITED COUNT FOM
                fidelities = [
                    fidelity(final_state, fom_ket) ** 2
                    for fom_ket in figure_of_merit_kets
                ]
                figure_of_merit = sum(fidelities)
                print(f"figure_of_merit: {figure_of_merit}")
                # FIDELITY_NOISE = 0.00
                # figure_of_merit += np.random.normal(size=1) * FIDELITY_NOISE
                if figure_of_merit < 0:
                    figure_of_merit = 0
                elif figure_of_merit > 1:
                    figure_of_merit = 1

                figure_of_merit_over_repeats.append(figure_of_merit)

            return 1 - np.mean(figure_of_merit_over_repeats)

        output = np.array([get_figure_of_merit(input_) for input_ in inputs])
        print(f"func f completed in {time.time() - start_time:.3f}s, output: {output}")
        return output

    return f


if __name__ == '__main__':
    from qutip_job_handlers.crossing import get_ghz_crossing
    from job_handlers.jobs.bo_utils import get_domain, optimise
    from job_handlers.timer import timer

    # trigger_event("job_progress", value1="Job started", value2=job_id)

    protocol_timesteps = 3
    t = 1e-6
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
    Delta_limits = (-crossing, 1.5 * crossing)

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
        bo = optimise(f, domain, max_iter=max_iter, exploit_iter=exploit_iter, initial_design_numdata_factor=initial_design_numdata_factor)

    print("x_opt", bo.x_opt)
    print("fx_opt", bo.fx_opt)

    # trigger_event("job_progress", value1="Job ended", value2=job_id)
