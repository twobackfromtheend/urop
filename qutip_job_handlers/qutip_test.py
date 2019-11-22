from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from scipy.interpolate import interp1d

import interaction_constants
from job_handlers.timer import timer
from qubit_system.geometry import RegularLattice
from qutip_job_handlers.hamiltonian import get_hamiltonian
from qutip_job_handlers.plotter import plot
from qutip_job_handlers.qutip_states import CustomGHZState
from qutip_job_handlers.solver import solve
from windows.tukey_window import tukey

if __name__ == '__main__':
    N = 8
    t = 2e-6

    N_RYD = 50
    C6 = interaction_constants.get_C6(N_RYD)

    LATTICE_SPACING = 1.5e-6

    BO_results = [
        # (
        #     "8_1d_std_8667",
        #     [6.31780270e+08, 4.81041016e+08, 9.29765575e+08, 1.14639670e+09, 1.70538760e+09, 1.31745385e+09],
        #     (8,), [True, True, True, True, True, True, True, True]
        # ),
        # (
        #     "8_1d_std_2_9298",
        #     [1.21129102e+08, 3.83847373e+08, 1.12636302e+09, 1.73484377e+09, 1.39461118e+09, 1.26388886e+09],
        #     (8,), [True, True, True, True, True, True, True, True]
        # ),
        # (
        #     "8_1d_alt_7498",
        #     [12504944.76765425, 15366680.6544196, 10803619.56131169, 22865447.71314488, 20650519.28562828,
        #      9654141.86655318],
        #     (8,), [True, False, True, False, True, False, True, False]
        # ),
        # (
        #     "8_1d_alt_2_7996",
        #     [ 1034883.35720177, 10746002.32511696, 13138604.21549956, 12611089.34283306, 14807475.81352524, 12823830.46326383],
        #     (8,), [True, False, True, False, True, False, True, False]
        # ),
        # (
        #     "8_2d_std_6049",
        #     [1.59175109e+09, 4.40798493e+08, 8.22430687e+08, 1.52515077e+09, 2.72788764e+09, 2.08805395e+09],
        #     (4, 2), [True, True, True, True, True, True, True, True]
        # ),
        (
            "8_2d_alt_9997",
            [8.94101353e+07, 1.34436283e+08, 3.17347152e+07, 1.90844269e+08, 9.70544131e+07, 8.64859020e+07],
            (4, 2), [True, False, False, True, True, False, False, True]
        )
    ]
    for name, optimised_params, shape, single_component in BO_results:
        geometry = RegularLattice(shape=shape, spacing=LATTICE_SPACING)
        ghz_state = CustomGHZState(N, single_component)

        timesteps = 3000
        t_list = np.linspace(0, t, timesteps)

        num_params = int(len(optimised_params) / 2)
        Omega_params = optimised_params[:num_params]
        Delta_params = optimised_params[num_params:]
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


        psi_0 = tensor([basis(2, 1) for _ in range(N)])
        hamiltonian = get_hamiltonian(N, C6, geometry, Omega, Delta)

        with timer("Solving system"):
            solve_result = solve(hamiltonian, psi_0, t_list)

        ghz_fidelity = fidelity(solve_result.states[-1], ghz_state.get_state_tensor()) ** 2
        print(f"ghz fidelity: {ghz_fidelity}")
        ax = plt.gca()
        with timer("Plotting"):

            # plot_ghz_states_overlaps(ax, solve_result, ghz_state, t_list, with_antisymmetric_ghz=True)
            plot(solve_result, ghz_state, t_list, Omega, Delta, )
            # plt.savefig(f"bo_qutip_{name}_{ghz_fidelity:.4f}.png", dpi=300)
            plt.show()
