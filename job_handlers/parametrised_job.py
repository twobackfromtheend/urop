import os
import time
from typing import Callable, List, Tuple

import numpy as np
import quimb as q
from GPyOpt.methods import BayesianOptimization

import interaction_constants
from ifttt_webhook import trigger_event
from job_handlers.hamiltonian import SpinHamiltonian, QType
from job_handlers.solver import solve_with_protocol
from job_handlers.utils import get_crossing
from protocol_generator.base_protocol_generator import BaseProtocolGenerator
from protocol_generator.interpolation_pg import InterpolationPG
from qubit_system.geometry import *
from qubit_system.utils import states
from qubit_system.utils.ghz_states import *

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

LATTICE_SPACING = 1.5e-6

print(f"C6: {C6:.3e}")
characteristic_V = C6 / (LATTICE_SPACING ** 6)
print(f"Characteristic V: {characteristic_V:.3e} Hz")

LOCAL_JOB_ENVVARS = {
    'PBS_JOBID': 'LOCAL_JOB',
    'N': '8',
    'Q_GEOMETRY': 'NoisyRegularLattice(shape=(4, 2), spacing=LATTICE_SPACING, spacing_noise=0e-9)',
    'Q_GHZ_STATE': 'CustomGHZState(N, [True, True, True, True, True, True, True, True])',
    'REPEATS': '1',
    'BO_BATCH_SIZE': '3',
    'BO_MAX_ITER': '50',
}

IS_LOCAL_JOB = bool(os.getenv("PBS_JOBID"))


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

batch_size = int(getenv("BO_BATCH_SIZE"))
max_iter = int(getenv("BO_MAX_ITER"))

print(
    "Parameters:\n"
    f"\tjob_id: {job_id}\n"
    f"\tN: {N}\n"
    f"\tQ_GEOMETRY: {geometry} ({geometry_envvar})\n"
    f"\tQ_GHZ_STATE: {ghz_state} ({ghz_state_envvar})\n"
    f"\tBO_BATCH_SIZE: {batch_size}\n"
    f"\tBO_MAX_ITER: {max_iter}\n"
)


def get_f(spin_ham: SpinHamiltonian, V: float, geometry: BaseGeometry,
          t_list: np.ndarray, psi_0: QType,
          ghz_state: BaseGHZState,
          protocol_generator: BaseProtocolGenerator):
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
            ghz_fidelity = q.fidelity(final_state, ghz_state.get_state_tensor())
            print(f"fidelity: {ghz_fidelity:.3f} (for protocol Omega: {Omega}, Delta: {Delta})")
            return 1 - ghz_fidelity

        output = np.array([get_figure_of_merit(input_) for input_ in inputs])
        print(f"func f completed in {time.time() - start_time:.3f}, output: {output}")
        return output

    return f


def get_domain(Omega_limits: Tuple[float, float], Delta_limits: Tuple[float, float], timesteps: int) -> List[dict]:
    return [
               {
                   'name': f'var_{i}',
                   'type': 'continuous',
                   'domain': Omega_limits
               }
               for i in range(timesteps)
           ] + [
               {
                   'name': f'var_{i}',
                   'type': 'continuous',
                   'domain': Delta_limits
               }
               for i in range(timesteps)
           ]


def optimise(f: Callable, domain: List[dict]):
    """
    :param f:
        function to optimize.
        It should take 2-dimensional numpy arrays as input and return 2-dimensional outputs (one evaluation per row).
    :param domain:
        list of dictionaries containing the description of the inputs variables
        (See GPyOpt.core.task.space.Design_space class for details).
    :return:
    """
    bo_kwargs = {
        'domain': domain,  # box-constraints of the problem
        'acquisition_type': 'EI',  # Selects the Expected improvement
        'initial_design_numdata': 4 * len(domain),  # Number of initial points
        'exact_feval': True
    }
    print(f"bo_kwargs: {bo_kwargs}")

    bo = BayesianOptimization(
        f=f,
        batch_size=batch_size,
        num_cores=batch_size,
        **bo_kwargs
    )

    optimisation_kwargs = {
        'max_iter': max_iter,
        # 'max_time': 300,
    }
    print(f"optimisation_kwargs: {optimisation_kwargs}")
    bo.run_optimization(**optimisation_kwargs)

    print(f"Optimised result: {bo.fx_opt}")
    print(f"Optimised controls: {bo.x_opt}")
    return bo


if __name__ == '__main__':
    trigger_event("job_progress", value1="Job started", value2=job_id)

    timesteps = 3
    t = 2e-6
    t_list = np.linspace(0, t, timesteps + 1)

    spin_ham = SpinHamiltonian.load(N)

    crossing = get_crossing(
        spin_ham=spin_ham, characteristic_V=characteristic_V,
        ghz_state=ghz_state, geometry=geometry,
        V=C6
    )
    Omega_limits = (0, crossing)
    Delta_limits = (0.5 * crossing, 1.5 * crossing)
    domain = get_domain(Omega_limits, Delta_limits, timesteps)

    protocol_generator = InterpolationPG(t_list, kind="cubic")

    f = get_f(
        spin_ham=spin_ham,
        V=C6,
        geometry=geometry,
        t_list=t_list,
        psi_0=q.kron(*states.get_ground_states(N)),
        ghz_state=ghz_state,
        protocol_generator=protocol_generator,
    )
    bo = optimise(f, domain)

    print("x_opt", bo.x_opt)
    print("fx_opt", bo.fx_opt)

    trigger_event("job_progress", value1="Job ended", value2=job_id)
