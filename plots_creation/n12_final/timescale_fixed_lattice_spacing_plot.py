import interaction_constants
from job_handlers.hamiltonian import SpinHamiltonian
from job_handlers.jobs.job_utils import get_geometry_and_ghz_state
import numpy as np
from typing import Tuple, Callable
import quimb as q
from numba import jit

from qubit_system.geometry import RegularLattice
from qubit_system.utils import states

from job_handlers.solver import solve_with_protocol

protocols = {
    "alt": {
        0.1: [2.03386323e+07, 1.41737622e+07, 2.01398813e+08, 2.03208585e+08],
        0.5: [8.35571487e+07, 1.02247973e+08, 1.94107312e+08, 1.83390663e+08],
        1: [1.57806740e+08, 1.83786685e+08, 1.96230522e+08, 1.74213095e+08],
        1.5: [2.42776483e+08, 2.16698779e+08, 2.04722874e+08, 1.71732265e+08],
        2: [2.92026840e+08, 3.46688336e+08, 1.89996239e+08, 1.75446712e+08],
        5: [8.32445646e+08, 9.34283424e+08, 1.91774646e+08, 1.78923640e+08],
        10: [1.97481739e+09, 1.75626655e+09, 2.04803046e+08, 1.75486098e+08],
    },
    "std": {
        0.1: [1.54866626e+08, 1.37735299e+08, 2.14156206e+09, 2.19712811e+09],
        0.5: [5.89219673e+08, 8.11220031e+08, 1.99975980e+09, 2.28307008e+09],
        1: [1.18709978e+09, 1.54390083e+09, 2.19045860e+09, 2.15510224e+09],
        1.5: [2.19919935e+09, 1.64303313e+09, 2.27127200e+09, 2.16709919e+09],
        2: [2.90620536e+09, 2.52747863e+09, 2.28972963e+09, 2.15799930e+09],
        5: [7.04737936e+09, 8.18506566e+09, 2.17843498e+09, 2.08034523e+09],
    }
}


def get_protocol(input_: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    slope = 1e9 / 1e-6  # Hz / s
    Omega_1, Omega_2, Delta_1, Delta_2 = input_

    ramp_up_duration = Omega_1 / slope
    ramp_down_duration = Omega_2 / slope
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


@jit(nopython=True)
def get_figure_of_merit(state, state_index_1: int, state_index_2: int):
    density_matrix = state @ state.conjugate().transpose()
    rho_00 = density_matrix[state_index_1, state_index_1].real
    rho_11 = density_matrix[state_index_2, state_index_2].real
    off_diag_1 = density_matrix[state_index_2, state_index_1]
    off_diagonal = 2 * np.abs(off_diag_1).real
    result = (rho_00 + rho_11 + off_diagonal) / 2
    return result, rho_00, rho_11, off_diagonal


N = 12

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)
print(f"C6: {C6:.3e}")

spin_ham = SpinHamiltonian.load(N)
lattice_spacing = 1.5e-6
print(f"\nLattice Spacing: {lattice_spacing:.5e}")
characteristic_V = C6 / (lattice_spacing ** 6)
print(f"Characteristic V: {characteristic_V:.5e} Hz")

for ghz in ['std', 'alt']:
    _geometry, ghz_state = get_geometry_and_ghz_state('', f"12_2d_{ghz}")

    for g, protocol in protocols[ghz].items():
        print(f"\ng: {g:.3f}")

        Omega, Delta, t_list = get_protocol(protocol)
        geometry = RegularLattice(shape=_geometry.shape, spacing=lattice_spacing)

        final_state = solve_with_protocol(
            spin_ham, V=C6, geometry=geometry, t_list=t_list, psi_0=q.kron(*states.get_ground_states(N)),
            Omega=Omega, Delta=Delta
        )
        ghz_state_tensor = ghz_state.get_state_tensor()
        fidelity = q.fidelity(final_state, ghz_state_tensor)
        print(f"Fidelity: {fidelity}")
        ghz_components = ghz_state._get_components()
        state_index_1 = ghz_components[0].argmax()
        state_index_2 = ghz_components[1].argmax()
        figure_of_merit, rho_00, rho_11, off_diagonal = get_figure_of_merit(final_state, state_index_1, state_index_2)
        print(f"Figure of Merit:")
        print(f"\tFoM {figure_of_merit}")
        print(f"\tr_0 {rho_00}")
        print(f"\tr_1 {rho_11}")
        print(f"\tdia {off_diagonal}")
    spin_ham.reset_geometry()
