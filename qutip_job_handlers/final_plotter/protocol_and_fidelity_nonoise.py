from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from qutip import tensor, basis
from tqdm import trange

import interaction_constants
from job_handlers.timer import timer
from qubit_system.geometry import *
from qutip_job_handlers import qutip_utils
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


def _plot_protocol_and_fidelity(ax1: Axes, ax2: Axes,
                                N: int,
                                t_list: np.ndarray, input_: np.ndarray,
                                solve_result,
                                first: bool, last: bool):
    ax1, ax2 = ax2, ax1

    mean_Omega, mean_Delta, Omega_lims, Delta_lims = get_protocol_noise(input_, t_list)

    t_list = t_list * 1e6

    Omega_color = "C0"
    Delta_color = "C3"
    Omega_Delta_lw = 1.5

    use_delta_ax = False
    if use_delta_ax:
        scaling = 1e9
        ls = "-"
        ax1.plot(t_list, mean_Omega / 1e9, color=Omega_color, lw=Omega_Delta_lw, alpha=0.8, ls=ls)
        ax1.fill_between(
            t_list,
            Omega_lims[0] / scaling,
            Omega_lims[1] / scaling,
            color=Omega_color, alpha=0.3
        )
        ax1.locator_params(nbins=3, axis='x')

        Delta_ax = ax1.twinx()
        Delta_ax.plot(t_list, mean_Delta / 1e9, color=Delta_color, lw=Omega_Delta_lw, alpha=0.8, ls=ls)
        Delta_ax.fill_between(
            t_list,
            Delta_lims[0] / scaling,
            Delta_lims[1] / scaling,
            color=Delta_color, alpha=0.3
        )

        # ax1.yaxis.set_major_formatter(ticker.EngFormatter('Hz'))
        ax1.locator_params(nbins=4, axis='y')
        if first:
            ax1.set_ylabel(r"$\Omega (t)$ [GHz]", color=Omega_color)
        ax1.yaxis.label.set_color(Omega_color)
        ax1.tick_params(axis='y', labelcolor=Omega_color)
        # Delta_ax.yaxis.set_major_formatter(ticker.EngFormatter('Hz'))
        Delta_ax.locator_params(nbins=4, axis='y')
        if last:
            Delta_ax.set_ylabel(r"$\Delta (t)$ [GHz]", color=Delta_color)
        Delta_ax.yaxis.label.set_color(Delta_color)
        Delta_ax.tick_params(axis='y', labelcolor=Delta_color)
    else:
        ls = "-"
        scaling = 1e9
        ax1.plot(t_list, mean_Omega / scaling, color=Omega_color, lw=Omega_Delta_lw, alpha=0.8, ls=ls,
                 label=r"$\Omega$")
        ax1.fill_between(
            t_list,
            Omega_lims[0] / scaling,
            Omega_lims[1] / scaling,
            color=Omega_color, alpha=0.3
        )
        ax1.locator_params(nbins=3, axis='x')

        ax1.plot(t_list, mean_Delta / scaling, color=Delta_color, lw=Omega_Delta_lw, alpha=0.8, ls=ls,
                 label=r"$\Delta$")
        ax1.fill_between(
            t_list,
            Delta_lims[0] / scaling,
            Delta_lims[1] / scaling,
            color=Delta_color, alpha=0.3
        )

    delta = (t_list.max() - t_list.min()) * 0.01
    ax1.set_xlim((t_list.min() - delta, t_list.max() + delta))
    ax1.grid(axis='x')

    # Panel 2
    cmap = plt.get_cmap("tab10")

    for i in trange(N + 1):
        # label = r"$\boldsymbol{\mathcal{F}_{" + str(
        #     i) + "}}$" if i == NUMBER_OF_EXCITED_WANTED else r"$\mathcal{F}_{" + str(i) + "}$"
        label = r"$\mathcal{F}_{" + str(i) + "}$"
        figure_of_merit_func = _get_f_for_excited_count(i)
        ax2.plot(
            t_list,
            [figure_of_merit_func(_instantaneous_state)
             for _instantaneous_state in solve_result.states],
            label=label,
            lw=2,
            alpha=0.8,
            color=cmap(i / 10)
        )
    if first:
        ax2.set_ylabel(r"$\mathcal{F}$")


    ax2.set_ylim((-0.1, 1.1))
    ax2.yaxis.set_ticks([0, 0.5, 1])
    ax2.grid()

    # box = ax2.get_position()
    # ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # ax2.legend(loc=9, ncol=2, framealpha=1)
    # ax2.legend(loc=9, ncol=2, fontsize="x-small")
    if len(geometry.shape) == 1:
        ax1.legend(numpoints=1, fontsize="x-small")


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
            'geometry': RegularLattice(shape=(9,), spacing=LATTICE_SPACING)
        },
        {
            'N': 9, 'NUMBER_OF_EXCITED_WANTED': 5,
            'protocol': [1.53246616e+09, 1.35082403e+06, 1.66224646e+09, -1.41329525e+09, -5.16672129e+08,
                         1.21000427e+09],
            'geometry': RegularLattice(shape=(3, 3), spacing=LATTICE_SPACING)
        },
        {
            'N': 8, 'NUMBER_OF_EXCITED_WANTED': 4,
            'protocol': [4.64808064e+08, 1.17036337e+09, 1.39071766e+09, -1.80894216e+09, 3.76733066e+08,
                         1.40485028e+09],
            'geometry': RegularLattice(shape=(2, 2, 2), spacing=LATTICE_SPACING)
        },
    ]:
        N = optimised_setup['N']
        NUMBER_OF_EXCITED_WANTED = optimised_setup['NUMBER_OF_EXCITED_WANTED']
        protocol = optimised_setup['protocol']
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


        # Setup protocol
        Omega_params = input_[:protocol_timesteps]
        Delta_params = input_[protocol_timesteps:]

        with timer(f"Creating QutipSpinHam (N={N})"):
            spin_ham = QutipSpinHamiltonian(N)

        Omega, Delta = get_protocol_from_input(input_, t_list)

        psi_0 = tensor([basis(2, 1) for _ in range(N)])

        hamiltonian = spin_ham.get_hamiltonian(C6, geometry, Omega, Delta)
        solve_result = solve(hamiltonian, psi_0, t_list)

        final_state = solve_result.final_state

        final_figure_of_merit_func = _get_f_for_excited_count(NUMBER_OF_EXCITED_WANTED)
        final_figure_of_merit = final_figure_of_merit_func(final_state)

        # Plotting
        gridspec_kwargs = {
            'top': 0.95,
            'bottom': 0.08,
            # With delta ax
            # 'left': 0.15,
            # 'right': 0.80,

            # Without delta ax
            'left': 0.15,
            'right': 0.96,
            'hspace': 0
        }
        gridspec_kwargs = {
            'top': 0.95,
            'bottom': 0.15,
            # With delta ax
            # 'left': 0.15,
            # 'right': 0.80,

            # Without delta ax
            'left': 0.25,
            'right': 0.96,
            'hspace': 0
        }
        # fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', gridspec_kw=gridspec_kwargs, figsize=(3.5, 3))
        # fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', gridspec_kw=gridspec_kwargs, figsize=(10, 9))

        plt.rcParams['font.size'] = 16
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', gridspec_kw=gridspec_kwargs, figsize=(3, 2.5))


        _plot_protocol_and_fidelity(ax1, ax2, N, t_list, input_, solve_result,
                                    first=True, last=True)
        # plt.savefig(f"bo_paper_plots/N{N}_OPT{NUMBER_OF_EXCITED_WANTED}_{len(geometry.shape)}d_legend.png",
        #             dpi=900)
        plt.savefig(f"bo_paper_plots/N{N}_OPT{NUMBER_OF_EXCITED_WANTED}_{len(geometry.shape)}d.png",
                    dpi=900)
        plt.show()
