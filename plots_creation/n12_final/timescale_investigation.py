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


def get_f(spin_ham: SpinHamiltonian, V: float, geometry: BaseGeometry,
          t_list: np.ndarray, psi_0: QType,
          ghz_state: BaseGHZState,
          protocol_generator: BaseProtocolGenerator):
    ghz_state_tensor = ghz_state.get_state_tensor()
    ghz_components = ghz_state._get_components()
    state_index_1 = ghz_components[0].argmax()
    state_index_2 = ghz_components[1].argmax()

    def f(input_: np.ndarray) -> float:
        with timer("Solving with protocol"):
            Omega, Delta = protocol_generator.get_protocol(input_)
            final_state = solve_with_protocol(
                spin_ham, V=V, geometry=geometry, t_list=t_list, psi_0=psi_0,
                Omega=Omega, Delta=Delta
            )
        figure_of_merit = get_figure_of_merit(final_state, state_index_1, state_index_2)
        # ghz_fidelity = q.fidelity(final_state, ghz_state_tensor)
        print(f"figure_of_merit: {figure_of_merit:.3f} for input: {input_}")

        output = figure_of_merit
        gc.collect()
        return output

    return f


if __name__ == '__main__':
    N = 12
    t = 1e-6

    N_RYD = 50
    C6 = interaction_constants.get_C6(N_RYD)
    print(f"C6: {C6:.3e}")
    _geometry, ghz_state = get_geometry_and_ghz_state('', os.getenv('Q_GHZ_STATE'))
    interpolation_timesteps = 3000
    t_list = np.linspace(0, t, interpolation_timesteps + 1)

    spin_ham = SpinHamiltonian.load(N)
    for lattice_spacing in np.linspace(1e-6, 4e-6, 5):
        print(f"\nLattice Spacing: {lattice_spacing:.3e}")
        characteristic_V = C6 / (lattice_spacing ** 6)
        print(f"Characteristic V: {characteristic_V:.3e} Hz")

        geometry = RegularLattice(shape=_geometry.shape, spacing=lattice_spacing)
        crossing = get_ghz_crossing(
            spin_ham=spin_ham, characteristic_V=characteristic_V,
            ghz_state=ghz_state, geometry=geometry,
            V=C6
        )

        for g in [0.1, 0.5, 0.8, 1, 1.2, 1.5, 2, 5, 10]:
            # g = max Omega / V0

            spin_ham.reset_geometry()

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

            Omega = g * crossing
            Delta = crossing

            fom = f(np.array([Omega] * 3 + [Delta] * 3))
            # print(f"Crossing: {crossing:.3e}")
