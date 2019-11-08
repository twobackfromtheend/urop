import os
import time
from functools import partial
from typing import Callable, List, Tuple

import numpy as np
import quimb as q
from GPyOpt.methods import BayesianOptimization
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey

import interaction_constants
from ifttt_webhook import trigger_event
from qubit_system.geometry import *
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem
from qubit_system.qubit_systems.static_qubit_system import StaticQubitSystem
from qubit_system.utils.ghz_states import *

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

LATTICE_SPACING = 1.5e-6

print(f"C6: {C6:.3e}")
characteristic_V = C6 / (LATTICE_SPACING ** 6)
print(f"Characteristic V: {characteristic_V:.3e} Hz")

job_id = os.getenv("PBS_JOBID")
N = int(os.getenv("N"))
geometry_envvar = os.getenv("Q_GEOMETRY")
geometry = eval(geometry_envvar)
ghz_state_envvar = os.getenv("Q_GHZ_STATE")
ghz_state = eval(ghz_state_envvar)
repeats = int(os.getenv("REPEATS"))
batch_size = int(os.getenv("BATCH_SIZE"))

print(
    "Parameters:\n"
    f"\tjob_id: {job_id}\n"
    f"\tN: {N}\n"
    f"\tQ_GEOMETRY: {geometry} ({geometry_envvar})\n"
    f"\tQ_GHZ_STATE: {ghz_state} ({ghz_state_envvar})\n"
    f"\tREPEATS: {repeats}\n"
)

trigger_event("job_progress", value1="Job started", value2=job_id)


def get_solved_episode(input_: np.ndarray,
                       N: int, V: float, geometry: BaseGeometry, t_list: np.ndarray,
                       ghz_state: BaseGHZState,
                       interpolation_timesteps: int = 3000) -> EvolvingQubitSystem:
    timesteps = len(t_list) - 1

    Omega_params = input_[:timesteps]
    Delta_params = input_[timesteps:]

    _t_list = np.linspace(0, t_list[-1], interpolation_timesteps + 1)
    interp = partial(interp1d, kind="quadratic", )

    Omega_func: Callable[[float], float] = interp(t_list, np.hstack((Omega_params, Omega_params[-1])))
    Omega_shape_window = tukey(interpolation_timesteps + 1, alpha=0.2)
    Omega = np.array([Omega_func(_t) * Omega_shape_window[_i] for _i, _t in enumerate(_t_list[:-1])])

    Delta_func: Callable[[float], float] = interp(t_list, np.hstack((Delta_params, Delta_params[-1])))
    Delta = np.array([Delta_func(_t) for _t in _t_list[:-1]])

    e_qs = EvolvingQubitSystem(
        N, V, geometry,
        Omega, Delta,
        _t_list,
        ghz_state=ghz_state
    )
    start_time = time.time()
    e_qs.solve()
    print(f"Solved in {time.time() - start_time:.3f}s")
    return e_qs


def get_f(N: int, V: float, geometry: BaseGeometry, t_list: np.ndarray, ghz_state: BaseGHZState):
    def f(inputs: np.ndarray) -> np.ndarray:
        """
        :param inputs: 2-dimensional array
        :return: 2-dimentional array, one-evaluation per row
        """
        print(f"inputs: {inputs}")

        def get_figure_of_merit(*args):
            e_qs = get_solved_episode(*args, N=N, V=V, geometry=geometry, t_list=t_list, ghz_state=ghz_state)
            # return 1 - e_qs.get_fidelity_with("ghz")
            # component_products = e_qs.get_fidelity_with("ground") * e_qs.get_fidelity_with("excited") * 4
            component_products = e_qs.get_fidelity_with("ghz_component_1") * e_qs.get_fidelity_with(
                "ghz_component_2") * 4
            # return 1 - component_products
            ghz_fidelity = e_qs.get_fidelity_with("ghz")
            ghz_above_half = max(ghz_fidelity - 0.5, 0) * 2
            print(f"fidelity: {ghz_fidelity:.3f}")
            return 1 - component_products * ghz_above_half

        return np.apply_along_axis(get_figure_of_merit, 1, inputs).reshape((-1, 1))

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
        'max_iter': 50,
        # 'max_time': 300,
    }
    print(f"optimisation_kwargs: {optimisation_kwargs}")
    bo.run_optimization(**optimisation_kwargs)

    print(f"Optimised result: {bo.fx_opt}")
    print(f"Optimised controls: {bo.x_opt}")
    return bo


def plot_result(bo: BayesianOptimization,
                N: int, V: float, geometry: BaseGeometry, t_list: np.ndarray, ghz_state: BaseGHZState,
                **kwargs):
    # bo.plot_convergence()
    optimised_controls = bo.x_opt
    e_qs = get_solved_episode(input_=optimised_controls, N=N, V=V, geometry=geometry, t_list=t_list,
                              ghz_state=ghz_state,
                              interpolation_timesteps=3000)
    print(f"fidelity: {e_qs.get_fidelity_with('ghz')}")
    plot_kwargs = {'show': True}
    e_qs.plot(**{**plot_kwargs, **kwargs})


def get_crossing(ghz_state: BaseGHZState, geometry: BaseGeometry, N: int, V: float):
    ghz_1 = ghz_state._get_components()[1]
    Delta_range = np.linspace(-characteristic_V * 5, characteristic_V * 5, 50)

    s_qs = StaticQubitSystem(
        N=N, V=V, geometry=geometry,
        Omega=0, Delta=Delta_range,
    )

    energies = []
    for detuning in Delta_range:
        H = s_qs.get_hamiltonian(detuning)
        energy = q.expec(H, ghz_1).real
        energies.append(energy)
    energies = np.array(energies)

    def find_root(x: np.ndarray, y: np.ndarray):
        """
        Finds crossing (where y equals 0), given that x, y is roughly linear.
        """
        if (y == 0).all():
            return np.nan
        _right_bound = (y < 0).argmax()
        _left_bound = _right_bound - 1
        crossing = y[_left_bound] / (y[_left_bound] - y[_right_bound]) \
                   * (x[_right_bound] - x[_left_bound]) + x[_left_bound]
        return crossing

    crossing = find_root(Delta_range, energies)
    print(f"Found crossing: {crossing:.3e}")

    return crossing


timesteps = 3
t = 2e-6
t_list = np.linspace(0, t, timesteps + 1)

crossing = get_crossing(ghz_state, geometry, N, C6)
Omega_limits = (0, crossing)
Delta_limits = (0.5 * crossing, 1.5 * crossing)
domain = get_domain(Omega_limits, Delta_limits, timesteps)

fidelities = []
for repeat in range(repeats):
    f = get_f(
        N, C6,
        geometry=geometry,
        t_list=t_list,
        ghz_state=ghz_state
    )
    bo = optimise(f, domain)
    optimised_controls = bo.x_opt
    e_qs = get_solved_episode(input_=optimised_controls, N=N, V=C6, geometry=geometry, t_list=t_list,
                              ghz_state=ghz_state,
                              interpolation_timesteps=3000)
    fidelity = e_qs.get_fidelity_with('ghz')
    # print(f"fidelity: {fidelity}")
    plot_result(bo, N, C6, geometry, t_list, ghz_state, show=True)
    # plot_result(bo, N, C6, geometry, t_list, ghz_state, savefig_name=f"bo_demo_{name}_{repeat}", show=False)
    fidelities.append(fidelity)

fidelities = np.array(fidelities)
print(f"\n\nfidelities:"
      f"\n\t {fidelities}"
      f"\n\t {fidelities.mean()}"
      f"\n\t {fidelities.std()}")

trigger_event("job_progress", value1="Job ended", value2=job_id)
