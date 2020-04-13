import gc
import os
import time
from functools import partial
from typing import Tuple, Callable

import numpy as np
import quimb as q
from numba import jit

import interaction_constants
from ifttt_webhook import trigger_event
from job_handlers.crossing import get_ghz_crossing
from job_handlers.hamiltonian import SpinHamiltonian, QType
from job_handlers.jobs.bo_utils import get_domain, optimise
from job_handlers.jobs.job_utils import get_geometry_and_ghz_state
from job_handlers.solver import solve_with_protocol
from job_handlers.timer import timer
from protocol_generator.base_protocol_generator import BaseProtocolGenerator
from protocol_generator.interpolation_pg import InterpolationPG
from protocol_generator.wide_interpolation_pg import WideInterpolationPG
from qubit_system.geometry import *
from qubit_system.utils import states
from qubit_system.utils.ghz_states import *


@jit(nopython=True)
def get_figure_of_merit(state, state_index_1: int, state_index_2: int):
    density_matrix = state @ state.conjugate().transpose()
    rho_00 = density_matrix[state_index_1, state_index_1].real
    rho_11 = density_matrix[state_index_2, state_index_2].real
    off_diag_1 = density_matrix[state_index_2, state_index_1]
    off_diagonal = 2 * np.abs(off_diag_1).real
    result = (rho_00 + rho_11 + off_diagonal) / 2
    return result


def get_f(
        spin_ham: SpinHamiltonian, V: float, geometry: BaseGeometry,
        psi_0: QType,
        ghz_state: BaseGHZState,
        get_protocol: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
):
    ghz_state_tensor = ghz_state.get_state_tensor()
    ghz_components = ghz_state._get_components()
    state_index_1 = ghz_components[0].argmax()
    state_index_2 = ghz_components[1].argmax()

    def f(inputs: np.ndarray) -> float:
        outputs = []
        for input_ in inputs:
            with timer("Solving with protocol"):
                Omega, Delta, t_list = get_protocol(input_)

                final_state = solve_with_protocol(
                    spin_ham, V=V, geometry=geometry, t_list=t_list, psi_0=psi_0,
                    Omega=Omega, Delta=Delta
                )
            # figure_of_merit = get_figure_of_merit(final_state, state_index_1, state_index_2)
            figure_of_merit = q.fidelity(final_state, ghz_state_tensor)
            print(f"figure_of_merit: {figure_of_merit:.3f} for input: {input_}")
            outputs.append(1 - figure_of_merit)
        gc.collect()
        return outputs

    return f


V_0_data = {
    # Geometry: (Number of interactions, At X * lattice spacing)
    '12_1d_alt': (1, 2),
    '12_2d_alt': (1, 2 ** 0.5),
    '12_3d_alt': (1, 2 ** 0.5),
    '12_1d_std': (1, 1),
    '12_2d_std': (1, 1),
    '12_3d_std': (1, 1),
}


def calculate_V_0(q_ghz_state: str, lattice_spacing: float):
    number_of_interactions, lattice_spacing_multiplier = V_0_data[q_ghz_state]
    return number_of_interactions * C6 / (lattice_spacing * lattice_spacing_multiplier) ** 6


if __name__ == '__main__':
    N = 12
    t = 1e-7

    N_RYD = 50
    C6 = interaction_constants.get_C6(N_RYD)
    print(f"C6: {C6:.3e}")
    q_ghz_state = os.getenv('Q_GHZ_STATE', '12_2d_std')
    print(f"q_ghz_state: {q_ghz_state}")
    _geometry, ghz_state = get_geometry_and_ghz_state('', q_ghz_state)

    q_slope_envvar = os.getenv('Q_SLOPE', '1e9 / 1e-6')  # Hz / s
    q_slope = eval(q_slope_envvar)
    print(f"q_slope: {q_slope} ({q_slope_envvar})")

    spin_ham = SpinHamiltonian.load(N)
    lattice_spacing = 1.5e-6
    print(f"\nLattice Spacing: {lattice_spacing:.5e}")
    characteristic_V = C6 / (lattice_spacing ** 6)
    print(f"Characteristic V: {characteristic_V:.5e} Hz")

    geometry = RegularLattice(shape=_geometry.shape, spacing=lattice_spacing)
    crossing = get_ghz_crossing(
        spin_ham=spin_ham, characteristic_V=characteristic_V,
        ghz_state=ghz_state, geometry=geometry,
        V=C6
    )

    V_0 = calculate_V_0(q_ghz_state, lattice_spacing)
    print(f'V_0: {V_0:.5e} Hz')


    def get_protocol(input_: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Omega_1, Omega_2, Delta_1, Delta_2 = input_

        ramp_up_duration = Omega_1 / q_slope
        ramp_down_duration = Omega_2 / q_slope
        between_duration = (ramp_up_duration + ramp_down_duration) / 2

        t = ramp_up_duration + ramp_down_duration + between_duration

        interpolation_timesteps = 3000
        t_list = np.linspace(0, t, interpolation_timesteps)

        Omega_ramp_up_index = int(ramp_up_duration / t * interpolation_timesteps)
        Omega_ramp_down_index = interpolation_timesteps - int(ramp_down_duration / t * interpolation_timesteps)
        Omega = np.concatenate([
            np.linspace(0, Omega_1, Omega_ramp_up_index),
            np.linspace(Omega_1, Omega_2, Omega_ramp_down_index - Omega_ramp_up_index),
            np.linspace(Omega_2, 0, interpolation_timesteps - Omega_ramp_down_index),
        ])
        Delta = np.linspace(Delta_1, Delta_2, interpolation_timesteps)

        print(f"Omega: {Omega.shape} (max: {Omega.max():.3e})")
        print(f"Delta: {Delta.shape}")
        print(f"t_list: {t_list.shape} (t: {t:.3e})")

        return Omega, Delta, t_list


    for g in [0.1, 0.5, 1, 1.5, 2, 5, 10]:
        print(f"\ng: {g:.3f}")
        # g = max Omega / V0

        Omega = g * V_0
        Delta = crossing
        print(f"\tOmega: {Omega:.3e}")
        print(f"\tDelta: {Delta:.3e}")

        spin_ham.reset_geometry()

        f = get_f(
            spin_ham=spin_ham,
            V=C6,
            geometry=geometry,
            psi_0=q.kron(*states.get_ground_states(N)),
            ghz_state=ghz_state,
            get_protocol=get_protocol,
        )

        blockade_radius = (C6 / (2 * Omega)) ** (1 / 6)
        print(f"Rydberg blockade radius: {blockade_radius :.3e}, ({blockade_radius / lattice_spacing:.3f} a)")

        lower_factor, upper_factor = (0.8, 1.2)
        Omega_limits = (Omega * lower_factor, Omega * upper_factor)

        Delta_limits = (crossing * 0.9, crossing * 1.1)
        domain = get_domain(Omega_limits, Delta_limits, 2)

        with timer(f"Optimising f"):
            bo = optimise(f, domain, max_iter=50, exploit_iter=0)

        print("x_opt", bo.x_opt)
        print("fx_opt", bo.fx_opt)
