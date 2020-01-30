from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from qutip import tensor, basis

import interaction_constants
from job_handlers.timer import timer
from qubit_system.geometry import *
from qutip_job_handlers.hamiltonian import QutipSpinHamiltonian
from qutip_job_handlers.qutip_bo_paper import get_protocol_from_input
from qutip_job_handlers.qutip_bo_paper_plotter_quimb_utils import get_f_function_generator
from qutip_job_handlers.solver import solve


def get_protocol_noise(input_: np.ndarray, t_list: np.ndarray):
    Omegas = []
    Deltas = []
    for i in range(100):
        Omega, Delta = get_protocol_from_input(input_, t_list)
        Omegas.append([Omega(_t) for _t in t_list])
        Deltas.append([Delta(_t) for _t in t_list])

    Omegas = np.array(Omegas)
    Deltas = np.array(Deltas)

    stats_axis = 0
    mean_Omega = Omegas.mean(axis=stats_axis)
    mean_Delta = Deltas.mean(axis=stats_axis)

    return mean_Omega, mean_Delta, \
           np.percentile(Omegas, [25, 75], axis=stats_axis), \
           np.percentile(Deltas, [25, 75], axis=stats_axis)


def protocol_and_fidelity_for_protocols(N: int, protocols: List[np.ndarray], t_list: np.ndarray,
                                        get_fidelity_noise: Callable):
    t_list = t_list * 1e6
    gridspec_kwargs = {
        'top': 0.95,
        'bottom': 0.15,
        'left': 0.25,
        'right': 0.96,
        'hspace': 0
    }
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', gridspec_kw=gridspec_kwargs, figsize=(3.5, 3))

    plt.rcParams['font.size'] = 16
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', gridspec_kw=gridspec_kwargs, figsize=(3, 2.5))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', gridspec_kw=gridspec_kwargs, figsize=(6, 5))

    ax1, ax2 = ax2, ax1

    Delta_ax = ax1

    for i, protocol in enumerate(protocols):
        OP = str(i + 1)

        input_ = np.array(protocol)

        # Plot Protocol
        mean_Omega, mean_Delta, Omega_lims, Delta_lims = get_protocol_noise(input_, t_list)
        # Omega_color = f"C{2 * i}"
        # Delta_color = f"C{2 * i + 1}"

        Omega_color = "C0"
        Delta_color = "C3"

        ls = '-' if i == 0 else '--'

        scaling = 1e9
        Omega_Delta_lw = 1.5
        ax1.plot(
            t_list, mean_Omega / scaling,
            color=Omega_color,
            lw=Omega_Delta_lw, alpha=0.8, ls=ls,
            label=rf"$\Omega$ (OP{OP})",
        )
        ax1.fill_between(
            t_list,
            Omega_lims[0] / scaling,
            Omega_lims[1] / scaling,
            color=Omega_color,
            alpha=0.3
        )
        ax1.locator_params(nbins=3, axis='x')

        Delta_ax.plot(
            t_list, mean_Delta / scaling,
            color=Delta_color,
            lw=Omega_Delta_lw, alpha=0.8, ls=ls,
            label=rf"$\Delta$ (OP{OP})",
        )
        Delta_ax.fill_between(
            t_list,
            Delta_lims[0] / scaling,
            Delta_lims[1] / scaling,
            color=Delta_color, alpha=0.3
        )

        # Panel 2
        mean_fidelities, fidelities_lims = get_fidelity_noise(input_)
        print("THINGTHINGTHING")
        print(mean_fidelities[-1])
        print("THINGTHINGTHING")
        ax2.plot(
            t_list,
            mean_fidelities,
            label=r"$\mathcal{F}$ using OP" + OP,
            lw=3,
            ls=ls,
            alpha=0.8,
            # color=cmap(i / N)
            color="C1"
        )
        # ax2.fill_between(
        #     t_list,
        #     fidelities_lims[0],
        #     fidelities_lims[1],
        #     alpha=0.3,
        #     # color=cmap(i / N),
        #     color="C1",
        # )

        ax1.locator_params(nbins=4, axis='y')
        # ax1.set_ylabel(r"$\Omega (t)$ [GHz]", color=Omega_color)
        # ax1.yaxis.label.set_color(Omega_color)
        # ax1.tick_params(axis='y', labelcolor=Omega_color)
        # Delta_ax.yaxis.set_major_formatter(ticker.EngFormatter('Hz'))
        # Delta_ax.locator_params(nbins=4, axis='y')
        # Delta_ax.set_ylabel(r"$\Delta (t)$ [GHz]", color=Delta_color)
        # Delta_ax.yaxis.label.set_color(Delta_color)
        # Delta_ax.tick_params(axis='y', labelcolor=Delta_color)

        delta = (t_list.max() - t_list.min()) * 0.01
        ax1.set_xlim((t_list.min() - delta, t_list.max() + delta))
        ax1.grid(axis='x')

        ax2.set_ylabel(r"$\mathcal{F}$")

        ax2.set_ylim((-0.1, 1.1))
        ax2.yaxis.set_ticks([0, 0.5, 1])
        ax2.grid()

        # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # ax1.legend()
        # ax2.legend()
        # if j == 0:
        #     ax1.legend()
    # if len(geometry.shape) == 1:
    #     ax2.legend()
    # ax1.legend(numpoints=1, framealpha=1, ncol=2,)
    # ax2.legend(numpoints=1, framealpha=1,)


if __name__ == '__main__':
    import plots_creation.utils

    N_RYD = 50
    C6 = interaction_constants.get_C6(N_RYD)
    LATTICE_SPACING = 1.5e-6

    for optimised_setup in [
        {
            'N': 9, 'NUMBER_OF_EXCITED_WANTED': 5,
            'protocol': [9.65565736e+08, 1.64856694e+08, 3.46018289e+08, -6.30177185e+08, -1.81838611e+08,
                         3.27368113e+08],
            'protocol_noise': [6.45405329e+08, 8.06989351e+08, 9.64930590e+08, -1.11032193e+09, 1.70090502e+08,
                               1.80252596e+09],
            'geometry': RegularLattice(shape=(9,), spacing=LATTICE_SPACING)
        },
        {
            'N': 9, 'NUMBER_OF_EXCITED_WANTED': 5,
            'protocol': [1.53246616e+09, 1.35082403e+06, 1.66224646e+09, -1.41329525e+09, -5.16672129e+08,
                         1.21000427e+09],
            'protocol_noise': [1.08818180e+09, 8.00003174e+08, 1.98909225e+09, -1.85038475e+09, 1.04629106e+09,
                               2.51910399e+09],
            'geometry': RegularLattice(shape=(3, 3), spacing=LATTICE_SPACING)
        },
        {
            'N': 8, 'NUMBER_OF_EXCITED_WANTED': 4,
            'protocol': [4.64808064e+08, 1.17036337e+09, 1.39071766e+09, -1.80894216e+09, 3.76733066e+08,
                         1.40485028e+09],
            # 'protocol_noise': [1.56731474e+09, 1.50590736e+09, 1.88557228e+08, 1.77865580e+09, -1.38031784e+08,
            #                    -8.13925779e+07],  # Optimised for 5 excited atoms
            'protocol_noise': [ 1.16704787e+09,  1.86138640e+09,  1.01750348e+09, -4.98809322e+08, -5.11296137e+08,  2.81992285e+09],
            'geometry': RegularLattice(shape=(2, 2, 2), spacing=LATTICE_SPACING)
        },
    ]:
        N = optimised_setup['N']
        NUMBER_OF_EXCITED_WANTED = optimised_setup['NUMBER_OF_EXCITED_WANTED']
        protocol = optimised_setup['protocol']
        protocol_noise = optimised_setup['protocol_noise']
        geometry = optimised_setup['geometry']

        protocol_timesteps = 3
        t = 1e-6
        timesteps = 500
        t_list = np.linspace(0, t, timesteps + 1)

        # Do not edit below
        input_ = np.array(protocol)

        # Setup figure of merit (excited count)
        _get_f_for_excited_count_quimb = get_f_function_generator(N)


        def _get_f_for_excited_count(count: int):
            def _get_f_excited_count(state):
                quimb_func = _get_f_for_excited_count_quimb(count)
                figure_of_merit = quimb_func(state)
                return figure_of_merit

            return _get_f_excited_count


        # Setup
        with timer(f"Creating QutipSpinHam (N={N})"):
            spin_ham = QutipSpinHamiltonian(N)

        psi_0 = tensor([basis(2, 1) for _ in range(N)])

        figure_of_merit_func = _get_f_for_excited_count(NUMBER_OF_EXCITED_WANTED)


        # Calculate F distribution
        def get_fidelity_noise(input_: np.ndarray):
            figure_of_merit_over_times = []
            for i in range(100):
            # for i in range(1):
                solve_result = do_iteration(input_)
                figure_of_merit_over_time = [
                    figure_of_merit_func(_instantaneous_state)
                    for _instantaneous_state in solve_result.states
                ]
                figure_of_merit_over_times.append(figure_of_merit_over_time)

            figure_of_merit_over_times = np.array(figure_of_merit_over_times)

            stats_axis = 0
            mean_figure_of_merit_over_times = figure_of_merit_over_times.mean(axis=stats_axis)

            return mean_figure_of_merit_over_times, \
                   np.percentile(figure_of_merit_over_times, [25, 75], axis=stats_axis)


        def do_iteration(input_: np.ndarray):
            Omega, Delta = get_protocol_from_input(input_, t_list)

            hamiltonian = spin_ham.get_hamiltonian(
                C6, geometry, Omega, Delta,
                filling_fraction=0.9,
            )
            solve_result = solve(hamiltonian, psi_0, t_list)

            final_state = solve_result.final_state
            final_figure_of_merit = figure_of_merit_func(final_state)
            print(final_figure_of_merit)
            return solve_result


        protocol_and_fidelity_for_protocols(N, [protocol, protocol_noise], t_list, get_fidelity_noise)
        plt.savefig(f"bo_paper_plots/N{N}_OPT{NUMBER_OF_EXCITED_WANTED}_{len(geometry.shape)}d_noisy.png",
                    dpi= 300)
        # plt.savefig(f"bo_paper_plots/N{N}_OPT{NUMBER_OF_EXCITED_WANTED}_{len(geometry.shape)}d_noisy_legend.png",
        #             dpi=900)
        # plt.show()
