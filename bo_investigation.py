import os
import pickle
import time
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import quimb as q
from GPyOpt.methods import BayesianOptimization
from quimb import entropy_subsys

import interaction_constants
from ifttt_webhook import trigger_event
from protocol_generator.base_protocol_generator import BaseProtocolGenerator
from protocol_generator.noisy_interpolation_pg import NoisyInterpolationPG
from qubit_system.geometry import *
from qubit_system.qubit_systems.evolving_qubit_system_with_noise import EvolvingQubitSystemWithGeometryNoise
from qubit_system.qubit_systems.static_qubit_system import StaticQubitSystem
from qubit_system.utils.ghz_states import *


np.set_printoptions(linewidth=500)

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

LATTICE_SPACING = 1.5e-6

print(f"C6: {C6:.3e}")
characteristic_V = C6 / (LATTICE_SPACING ** 6)
print(f"Characteristic V: {characteristic_V:.3e} Hz")

print(f"Lattice spacing: {LATTICE_SPACING}")

job_id = "BO_INVESTIGATION"
N = 8

geometry_envvar = "NoisyRegularLattice(shape=(4, 2), spacing=LATTICE_SPACING, spacing_noise=10e-9)"
geometry = eval(geometry_envvar)
# ghz_state_envvar = "CustomGHZState(N, [True, False, True, False, True, False, True, False])"
# ghz_state_envvar = "CustomGHZState(N, [True, True, True, True, True, True, True, True])"
ghz_state_envvar = "CustomGHZState(N, [True, False, False, True, True, False, False, True])"
ghz_state = eval(ghz_state_envvar)

repeats = 5
evaluation_iterations = 10

print(
    "Parameters:\n"
    f"\tjob_id: {job_id}\n"
    f"\tN: {N}\n"
    f"\tQ_GEOMETRY: {geometry} ({geometry_envvar})\n"
    f"\tQ_GHZ_STATE: {ghz_state} ({ghz_state_envvar})\n"
    f"\tREPEATS: {repeats}\n"
    f"\tEVALUATION_ITERATIONS: {evaluation_iterations}\n"
)

trigger_event("job_progress", value1="Job started", value2=job_id)


def get_f(e_qs: EvolvingQubitSystemWithGeometryNoise, protocol_generator: BaseProtocolGenerator):
    def f(inputs: np.ndarray) -> np.ndarray:
        """
        :param inputs: 2-dimensional array
        :return: 2-dimentional array, one-evaluation per row
        """
        print(f"inputs: {inputs}")
        start_time = time.time()

        def get_figure_of_merit(input_: np.ndarray):
            fidelities = []
            for i in range(5):
                Omega, Delta = protocol_generator.get_protocol(input_)
                e_qs.solve_with_protocol(Omega, Delta)
                # component_products = e_qs.get_fidelity_with("ground") * e_qs.get_fidelity_with("excited") * 4
                # component_products = e_qs.get_fidelity_with("ghz_component_1") * e_qs.get_fidelity_with(
                #     "ghz_component_2") * 4
                ghz_fidelity = e_qs.get_fidelity_with("ghz")
                # print(f"fidelity: {ghz_fidelity:.3f}")
                fidelities.append(ghz_fidelity)
            fidelities = np.array(fidelities)
            print(f"fidelities: {fidelities} ({fidelities.mean():.3f} pm {fidelities.std():.3f})")

            return 1 - fidelities.mean()
            # return 1 - component_products
            # ghz_above_half = max(ghz_fidelity - 0.5, 0) * 2
            # return 1 - component_products * ghz_above_half

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
        # 'initial_design_numdata': 4,  # Number of initial points
        'initial_design_numdata': 4 * len(domain),  # Number of initial points
        'exact_feval': False
    }
    print(f"bo_kwargs: {bo_kwargs}")

    bo = BayesianOptimization(
        f=f,
        # maximise=True,
        # initial_design_type='latin',
        # model_type="sparseGP",
        # batch_size=6,
        **bo_kwargs
    )

    optimisation_kwargs = {
        'max_iter': 300,
        # 'max_time': 300,
    }
    print(f"optimisation_kwargs: {optimisation_kwargs}")
    bo.run_optimization(**optimisation_kwargs)

    print(f"Optimised result: {bo.fx_opt}")
    print(f"Optimised controls: {bo.x_opt}")
    return bo


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
                   * (x[_right_bound] - x[_left_bound]) \
                   + x[_left_bound]
        return crossing

    crossing = find_root(Delta_range, energies)
    print(f"Found crossing: {crossing:.3e}")

    return crossing


protocol_timesteps = 3
t = 2e-6
interpolation_timesteps = 3000
t_list = np.linspace(0, t, interpolation_timesteps + 1)

crossing = get_crossing(ghz_state, geometry, N, C6)
Omega_limits = (0, crossing)
Delta_limits = (0.5 * crossing, 1.5 * crossing)
domain = get_domain(Omega_limits, Delta_limits, protocol_timesteps)

blockade_radius = (C6 / (2 * Omega_limits[1])) ** (1 / 6)
print(f"Rydberg blockade radius: {blockade_radius :.3e}, ({blockade_radius / LATTICE_SPACING:.3f} a)")

fidelities = []
for repeat in range(repeats):
    e_qs = EvolvingQubitSystemWithGeometryNoise(N, C6, geometry, t_list, ghz_state)
    protocol_generator = NoisyInterpolationPG(t_list, kind="quadratic", noise=0.01)
    f = get_f(e_qs, protocol_generator)

    bo = optimise(f, domain)

    optimised_controls = bo.x_opt

    evaluation_fidelities = []
    for i in range(evaluation_iterations):
        optimised_Omega, optimised_Delta = protocol_generator.get_protocol(optimised_controls)
        e_qs.solve_with_protocol(optimised_Omega, optimised_Delta)
        fidelity = e_qs.get_fidelity_with('ghz')
        evaluation_fidelities.append(fidelity)
    print(
        f"Evaluation fidelities: {evaluation_fidelities}\n"
        f" {np.array(evaluation_fidelities).mean():.3f} pm {np.array(evaluation_fidelities).std():.3f}"
    )

    fidelities.append(evaluation_fidelities)

    protocol_generator.noise = 0.  # Set noise to 0 for saving optimised protocol.
    optimised_protocol = protocol_generator.get_protocol(optimised_controls)

    save_file_path = Path(__file__).parent / f"{job_id}_{time.time()}.optimised.pkl"
    # with save_file_path.open("wb") as f:
    #     save_dict = {
    #         'N': N,
    #         'geometry_envvar': geometry_envvar,
    #         'ghz_state_envvar': ghz_state_envvar,
    #         't_list': t_list,
    #         'optimised_vars': optimised_controls,
    #         'optimised_protocol': optimised_protocol,
    #         'evaluation_fidelities': evaluation_fidelities
    #     }
    #     pickle.dump(save_dict, f)

fidelities = np.array(fidelities)
print(f"\n\nfidelities:"
      f"\n\t {fidelities}"
      f"\n\t {fidelities.mean()}"
      f"\n\t {fidelities.std()}")

trigger_event("job_progress", value1="Job ended", value2=job_id)
