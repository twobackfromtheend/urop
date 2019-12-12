from collections import defaultdict
from itertools import combinations, product
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

import plots_creation.utils

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)
LATTICE_SPACING = 1.5e-6

# N = 10
# geometry = RegularLattice(shape=([10]), spacing=LATTICE_SPACING)
# protocol = [1.79854680e+08, 1.02270560e+07, 6.33317516e+08, -2.94542650e+08, 7.42496734e+08, -1.19984168e+09]

# N = 8
# NUMBER_OF_EXCITED_WANTED = 4
# ONLY CONTROL FIELD NOISE
# protocol = [2.31816577e+08, 7.87273427e+08, 3.10767995e+08, -8.80444567e+08, 1.17103600e+09, 6.89902845e+08]
# geometry = RegularLattice(shape=([8]), spacing=LATTICE_SPACING)
# protocol = [2.17951022e+08, 1.17038310e+09, 2.46058217e+08, -1.01583192e+09, 1.87461499e+09, 9.54290034e+08]
# geometry = RegularLattice(shape=(4, 2), spacing=LATTICE_SPACING)
# protocol = [8.00951105e+08, 2.21634222e+09, 1.16533709e+09, -1.22497021e+09, 1.07092118e+09, 1.60474119e+09]
# protocol = [5.13662969e+08, 1.45432946e+09, 1.29034091e+09, -1.35371684e+09, 3.23240317e+09, 2.52558790e+09]
# protocol = [1.43474072e+09, 4.11560646e+08, 7.14812176e+08, -9.91129358e+08, 2.23143839e+09, 2.64585531e+09]
# protocol = [2.15610544e+09, 2.20735177e+09, 1.71524082e+09, -1.63810348e+09, 2.77942308e+09, 1.72520940e+09]
# protocol = [1.21164540e+09, 1.08624614e+09, 2.07362541e+09, -1.22074505e+09, 2.17430166e+09, 2.16417839e+09]
# protocol = [1.00373258e+09, 1.42238334e+09, 1.74782340e+09, -1.23725305e+09, -2.01987016e+09, 9.85208369e+08]
# geometry = RegularLattice(shape=(2, 2, 2), spacing=LATTICE_SPACING)

# WITH FIDELITY NOISE
# protocol = [2.79488211e+08, 1.02682632e+09, 4.17768930e+08, -1.02715406e+08, -8.46245111e+08, 1.74649154e+08]
# protocol = [4.26941153e+08, 4.54242969e+08, 4.63694399e+07, -7.39581885e+08, 8.42035048e+08, 2.67549651e+08]
# geometry = RegularLattice(shape=([8]), spacing=LATTICE_SPACING)
# protocol = [8.72605870e+08, 1.40417535e+09, 8.42870942e+08, -1.42483321e+09, 2.05160033e+09, 1.87079651e+09]
# protocol = [4.05560976e+08, 1.83260778e+09, 1.22206913e+09, -1.04226933e+09, -1.87024181e+08, 1.43798017e+09]
# protocol = [3.81320505e+08, 1.32584084e+09, 1.53935430e+09, -1.01526210e+09, 2.38888572e+09, 2.32190751e+09]
# protocol = [5.10491807e+08, 2.40518030e+08, 8.04044086e+08, -1.38067535e+09, 1.74286047e+08, 2.24987222e+09]
# geometry = RegularLattice(shape=(4, 2), spacing=LATTICE_SPACING)
# geometry = RegularLattice(shape=(2, 2, 2), spacing=LATTICE_SPACING)

# With fidelity noise, optimised for 3 excited
# NUMBER_OF_EXCITED_WANTED = 3
# protocol = [6.81401592e+08, 1.08468298e+09, 5.17823260e+08, 7.73973919e+08, -8.93365637e+08, 4.18852107e+08]
# geometry = RegularLattice(shape=([8]), spacing=LATTICE_SPACING)
# protocol = [1.40322277e+09, 1.66426921e+09, 1.66577682e+09, 2.30298907e+08, -9.04420800e+08, 7.98280817e+08]
# geometry = RegularLattice(shape=(4, 2), spacing=LATTICE_SPACING)
# protocol = [1.66936935e+09, 5.38748194e+07, 4.59525072e+08, 1.52937585e+09, 3.34735558e+09, -1.66241910e+09]
# geometry = RegularLattice(shape=(2, 2, 2), spacing=LATTICE_SPACING)


# With fidelity noise, optimised for 5 excited
# NUMBER_OF_EXCITED_WANTED = 5
# protocol = [9.41947077e+08, 2.01377061e+09, 1.77940440e+09, 3.89445853e+08, -2.01177870e+09, 2.47678715e+09]
# geometry = RegularLattice(shape=(2, 2, 2), spacing=LATTICE_SPACING)
# 50 iterations
# protocol = [5.43302529e+08, 1.39320866e+09, 2.25463689e+09, 3.07567638e+09, -1.87082158e+09, -1.14956397e+09]
# geometry = RegularLattice(shape=(2, 2, 2), spacing=LATTICE_SPACING)


# N = 8
# NUMBER_OF_EXCITED_WANTED = 3
# protocol = [3.40262902e+08, 1.87803597e+08, 7.48893605e+08, 4.22978061e+08, -3.81584270e+07, 4.82244774e+08]
# geometry = RegularLattice(shape=(8,), spacing=LATTICE_SPACING)


N = 8
# NUMBER_OF_EXCITED_WANTED = 5
# protocol = [4.71659374e+08, 1.52607372e+09, 1.70433758e+09, 1.81326929e+09, 2.36338135e+08, -2.85593335e+08]
NUMBER_OF_EXCITED_WANTED = 4
# protocol = [5.81388735e+08, 2.32204135e+09, 1.92599260e+09, -8.65853702e+08, -1.48619438e+09, 1.46206030e+09]
protocol = [4.64808064e+08, 1.17036337e+09, 1.39071766e+09, -1.80894216e+09, 3.76733066e+08, 1.40485028e+09]
geometry = RegularLattice(shape=(2, 2, 2), spacing=LATTICE_SPACING)

# N = 9
# NUMBER_OF_EXCITED_WANTED = 5
# protocol = [9.65565736e+08, 1.64856694e+08, 3.46018289e+08, -6.30177185e+08, -1.81838611e+08, 3.27368113e+08]
# geometry = RegularLattice(shape=(9,), spacing=LATTICE_SPACING)
# protocol = [1.53246616e+09, 1.35082403e+06, 1.66224646e+09, -1.41329525e+09, -5.16672129e+08, 1.21000427e+09]
# geometry = RegularLattice(shape=(3, 3), spacing=LATTICE_SPACING)

# N = 9
# # NUMBER_OF_EXCITED_WANTED = 5
# # protocol = [9.65565736e+08, 1.64856694e+08, 3.46018289e+08, -6.30177185e+08, -1.81838611e+08, 3.27368113e+08]
# NUMBER_OF_EXCITED_WANTED = 4
# # protocol = [1.30991746e+08, 1.22088796e+08, 4.10090912e+08, 7.19754220e+08, 7.05515133e+08, 2.84482008e+08]
# protocol = [8.65788393e+08, 1.21111696e+09, 1.09763099e+09, 8.07236002e+08, -9.90691682e+08, 1.61612169e+09]
# # NUMBER_OF_EXCITED_WANTED = 3
# # protocol = [9.46744815e+08, 2.69339934e+08, 6.99297013e+08, 5.51689614e+08, -2.88146208e+08, 1.29382414e+08]
#
# geometry = RegularLattice(shape=(9,), spacing=LATTICE_SPACING)


# FILLING FRACTION LATTICE
N = 9
NUMBER_OF_EXCITED_WANTED = 5
protocol = [4.71866199e+08, 6.65469914e+08, 1.07187077e+09, 8.28283448e+07, -5.93877163e+08, -1.02137811e+09]
geometry = RegularLattice(shape=(9,), spacing=LATTICE_SPACING)

protocol_timesteps = 3
t = 1e-6
timesteps = 500
t_list = np.linspace(0, t, timesteps + 1)

# Do not edit below
input_ = np.array(protocol)

# Setup figure of merit (excited count)
states_list = qutip_utils.get_states(N)
state_tensors_by_excited_count = defaultdict(list)
for state in states_list:
    state_label = qutip_utils.get_label_from_state(state)
    state_excited_count = sum(letter == "e" for letter in state_label)
    state_tensors_by_excited_count[state_excited_count].append(tensor(*state))

figure_of_merit_kets = state_tensors_by_excited_count[NUMBER_OF_EXCITED_WANTED]

_get_f_for_excited_count_quimb = get_f_function_generator(N)


def _get_f_for_excited_count(count: int):
    figure_of_merit_kets = state_tensors_by_excited_count[count]

    def _get_f_excited_count(state):
        # fidelities = [
        #     fidelity(state, fom_ket) ** 2
        #     for fom_ket in figure_of_merit_kets
        # ]
        # figure_of_merit_qutip = sum(fidelities)
        quimb_func = _get_f_for_excited_count_quimb(count)
        figure_of_merit = quimb_func(state)
        # print(f"{figure_of_merit_qutip:.5f}, {figure_of_merit:.5f}, {figure_of_merit - figure_of_merit_qutip:.5f}")
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

# print(len(solve_result.states))
final_state = solve_result.final_state

final_figure_of_merit_func = _get_f_for_excited_count(NUMBER_OF_EXCITED_WANTED)
final_figure_of_merit = final_figure_of_merit_func(final_state)
print(final_figure_of_merit)


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
           np.percentile(Omegas, [15.9, 84.1], axis=stats_axis), \
           np.percentile(Deltas, [15.9, 84.1], axis=stats_axis)


def _plot_protocol_and_fidelity(ax1: Axes, ax2: Axes,
                                t_list: np.ndarray,
                                solve_result,
                                first: bool, last: bool):
    ax1, ax2 = ax2, ax1
    ax1.xaxis.set_major_formatter(ticker.EngFormatter('s'))

    mean_Omega, mean_Delta, Omega_lims, Delta_lims = get_protocol_noise(input_, t_list)

    # Omega_array = np.array([Omega(_t) for _t in t_list])
    # Delta_array = np.array([Delta(_t) for _t in t_list])

    Omega_color = "C0"
    Delta_color = "C3"

    use_delta_ax = False
    if use_delta_ax:
        ls = "-"
        ax1.plot(t_list, mean_Omega / 1e9, color=Omega_color, lw=3, alpha=0.8, ls=ls)
        ax1.fill_between(t_list,
                         (mean_Omega - Omegas_std) / 1e9,
                         (mean_Omega + Omegas_std) / 1e9,
                         color=Omega_color, alpha=0.3)
        ax1.locator_params(nbins=3, axis='x')

        Delta_ax = ax1.twinx()
        Delta_ax.plot(t_list, mean_Delta / 1e9, color=Delta_color, lw=3, alpha=0.8, ls=ls)
        Delta_ax.fill_between(t_list,
                              (mean_Delta - Deltas_std) / 1e9,
                              (mean_Delta + Deltas_std) / 1e9,
                              color=Delta_color, alpha=0.3)

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
        factor = 1
        ax1.plot(t_list, mean_Omega / factor, color=Omega_color, lw=3, alpha=0.8, ls=ls, label=r"$\Omega$")
        ax1.fill_between(t_list,
                         (mean_Omega - Omegas_std) / factor,
                         (mean_Omega + Omegas_std) / factor,
                         color=Omega_color, alpha=0.3)
        ax1.locator_params(nbins=3, axis='x')

        ax1.plot(t_list, mean_Delta / factor, color=Delta_color, lw=3, alpha=0.8, ls=ls, label=r"$\Delta$")
        ax1.fill_between(t_list,
                         (mean_Delta - Deltas_std) / factor,
                         (mean_Delta + Deltas_std) / factor,
                         color=Delta_color, alpha=0.3)
        ax1.yaxis.set_major_formatter(ticker.EngFormatter('Hz'))

    delta = (t_list.max() - t_list.min()) * 0.01
    ax1.set_xlim((t_list.min() - delta, t_list.max() + delta))
    ax1.grid(axis='x')
    # Panel 2
    # cmap = plt.get_cmap("Set1")
    cmap = plt.get_cmap("tab10")
    # cmap = plt.get_cmap("Pastel1")
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
            lw=3,
            alpha=0.8,
            color=cmap(i / 10)
        )
    if first:
        # ax2.set_ylabel(r"Figure of merit $\mathcal{F}$")
        ax2.set_ylabel(r"$\mathcal{F}$")
    # ax2.legend(loc=2)

    ax2.set_ylim((-0.1, 1.1))
    ax2.yaxis.set_ticks([0, 0.5, 1])
    ax2.grid()

    # box = ax2.get_position()
    # ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # ax2.legend(loc=9, ncol=3, fontsize="x-small")
    # ax1.legend()


def _plot_atom_excited_population(ax1: Axes,
                                  t_list: np.ndarray,
                                  solve_result, ):
    with timer("Getting excited atom indices for each product basis state"):
        states = qutip_utils.get_states(N)

        product_basis_index_to_label = {}
        product_basis_index_to_excited_atom_indices = {}
        for state in states:
            label = qutip_utils.get_label_from_state(state)
            product_basis_index = qutip_utils.get_product_basis_states_index(state)
            product_basis_index_to_label[product_basis_index] = label
            excited_atom_indices = [i for i in range(N) if label[i] == 'e']
            product_basis_index_to_excited_atom_indices[product_basis_index] = excited_atom_indices

    with timer("Getting atom excited populations over time"):
        atom_populations_over_time = []
        for state in solve_result.states:
            atom_populations = [0 for _ in range(N)]
            state_data = state.data
            nonzeros = state_data.nonzero()[0]
            for product_basis_index in nonzeros:
                population = np.abs(state_data[product_basis_index, 0]) ** 2
                excited_atom_indices = product_basis_index_to_excited_atom_indices[product_basis_index]
                for i in excited_atom_indices:
                    atom_populations[i] += population
            # print(atom_populations)
            atom_populations_over_time.append(atom_populations)
        atom_populations_over_time = np.array(atom_populations_over_time).T

    COLORMAP = 'viridis'
    NORM = Normalize(vmin=0, vmax=1, clip=True)

    ax1.imshow(
        atom_populations_over_time,
        aspect="auto",
        cmap=COLORMAP, norm=NORM,
        extent=(0, t_list[-1], -0.5, N - 0.5)
    )
    ax1.set_ylabel("Atomic excited state population")

    mappable = ScalarMappable(norm=NORM, cmap=COLORMAP)
    # cbar = plt.colorbar(mappable, ax=ax1, ticks=[1e-3, 1e-2, 1e-1, 1], extend='min')
    # cbar = plt.colorbar(mappable, cax=ax1)
    axins = inset_axes(ax1,
                       width="5%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='center left',
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=ax1.transAxes,
                       borderpad=0,
                       )
    cbar = plt.colorbar(mappable, cax=axins)

    ax1.xaxis.set_major_formatter(ticker.EngFormatter('s'))

    ax1.set_yticks(np.arange(N + 1) - 0.5)
    ax1.grid()
    ax1.get_yaxis().set_ticklabels([])


def _plot_atom_excited_population_at_end(ax1: Axes, solve_result, ):
    with timer("Getting excited atom indices for each product basis state"):
        states = qutip_utils.get_states(N)

        product_basis_index_to_label = {}
        product_basis_index_to_excited_atom_indices = {}
        for state in states:
            label = qutip_utils.get_label_from_state(state)
            if label == 'eggegeeg':
                label = 'geegegge'
            product_basis_index = qutip_utils.get_product_basis_states_index(state)
            product_basis_index_to_label[product_basis_index] = label
            excited_atom_indices = [i for i in range(N) if label[i] == 'e']
            product_basis_index_to_excited_atom_indices[product_basis_index] = excited_atom_indices

    with timer("Getting atom excited populations at T"):
        state = solve_result.final_state
        atom_populations = [0 for _ in range(N)]
        state_data = state.data
        nonzeros = state_data.nonzero()[0]
        for product_basis_index in nonzeros:
            population = np.abs(state_data[product_basis_index, 0]) ** 2
            excited_atom_indices = product_basis_index_to_excited_atom_indices[product_basis_index]
            for i in excited_atom_indices:
                atom_populations[i] += population
        # print(atom_populations)

        atom_populations = np.array(atom_populations).T

    COLORMAP = 'viridis'
    # NORM = LogNorm(vmin=1e-3, vmax=1, clip=True)
    NORM = Normalize(vmin=0, vmax=1, clip=True)

    if len(geometry.shape) == 1:
        x_s = geometry.coordinates[:, 0]
        ax1.bar(x_s, atom_populations, width=LATTICE_SPACING / 3)
        ax1.set_xlim((-LATTICE_SPACING / 2, (N - 0.5) * LATTICE_SPACING))
        # ax1.set_ylim((-0.1, 1.1))
        ax1.set_ylim((0, 1))
        ax1.set_xticks(np.arange(N) * LATTICE_SPACING)
        ax1.get_xaxis().set_ticklabels(np.arange(N) + 1)
        ax1.grid(axis='y')
        # ax1.set_ylabel("Atomic excited state population")
        plt.tight_layout()
    elif len(geometry.shape) == 2:
        x_s = geometry.coordinates[:, 0]
        y_s = geometry.coordinates[:, 1]
        # print(plt.rcParams['lines.markersize'])
        ax1.scatter(
            x_s, y_s,
            c=atom_populations, cmap=COLORMAP, norm=NORM,
            s=300,
            edgecolors='k',
        )
        ax1.set_xlim((-LATTICE_SPACING / 2, x_s.max() + LATTICE_SPACING / 2))
        ax1.set_ylim((-LATTICE_SPACING / 2, y_s.max() + LATTICE_SPACING / 2))
        ax1.set_xticks(np.arange(geometry.shape[0]) * LATTICE_SPACING)
        ax1.get_xaxis().set_ticklabels(np.arange(geometry.shape[0]) + 1)
        ax1.set_yticks(np.arange(geometry.shape[1]) * LATTICE_SPACING)
        ax1.get_yaxis().set_ticklabels(np.arange(geometry.shape[1]) + 1)

        mappable = ScalarMappable(norm=NORM, cmap=COLORMAP)
        cbar = plt.colorbar(mappable, ax=ax1)
        plt.tight_layout()
        # ax1.grid(axis='y')
        # ax1.set_ylabel("Atomic excited state population")
    elif len(geometry.shape) == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = ax1.figure
        ax1.set_axis_off()
        ax3d = fig.add_subplot(111, projection='3d')
        x_s = geometry.coordinates[:, 0]
        y_s = geometry.coordinates[:, 1]
        z_s = geometry.coordinates[:, 2]

        ax3d.scatter(
            x_s, y_s, z_s,
            c=atom_populations, cmap=COLORMAP, norm=NORM,
            depthshade=False,
            s=300,
            edgecolors='k',
        )
        r = [0, LATTICE_SPACING]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s - e)) == r[1] - r[0]:
                ax3d.plot3D(*zip(s, e), color="darkgrey", alpha=0.8)

        ax3d.set_xlim((-LATTICE_SPACING / 2, x_s.max() + LATTICE_SPACING / 2))
        ax3d.set_ylim((-LATTICE_SPACING / 2, y_s.max() + LATTICE_SPACING / 2))
        ax3d.set_zlim((-LATTICE_SPACING / 2, z_s.max() + LATTICE_SPACING / 2))
        ax3d.set_xticks(np.arange(geometry.shape[0]) * LATTICE_SPACING)
        ax3d.get_xaxis().set_ticklabels(np.arange(geometry.shape[0]) + 1)
        ax3d.set_yticks(np.arange(geometry.shape[1]) * LATTICE_SPACING)
        ax3d.get_yaxis().set_ticklabels(np.arange(geometry.shape[1]) + 1)
        ax3d.set_zticks(np.arange(geometry.shape[2]) * LATTICE_SPACING)
        ax3d.get_zaxis().set_ticklabels(np.arange(geometry.shape[2]) + 1)

        mappable = ScalarMappable(norm=NORM, cmap=COLORMAP)
        cbar = plt.colorbar(mappable, ax=ax3d, shrink=0.7)

        plt.tight_layout()
        # ax1.grid(axis='y')
        # ax1.set_ylabel("Atomic excited state population")
    else:
        mappable = ScalarMappable(norm=NORM, cmap=COLORMAP)
        # cbar = plt.colorbar(mappable, ax=ax1, ticks=[1e-3, 1e-2, 1e-1, 1], extend='min')
        # cbar = plt.colorbar(mappable, cax=ax1)
        axins = inset_axes(ax1,
                           width="5%",  # width = 5% of parent_bbox width
                           height="100%",  # height : 50%
                           loc='center left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=ax1.transAxes,
                           borderpad=0,
                           )
        cbar = plt.colorbar(mappable, cax=axins)

    # ax1.xaxis.set_major_formatter(ticker.EngFormatter('s'))

    # ax1.set_yticks(np.arange(N + 1) - 0.5)
    # ax1.grid()
    # ax1.get_yaxis().set_ticklabels([])


def protocol_and_fidelity_for_protocols():
    protocols = {
        # "Without $\mathcal{F}$ noise":
        "without_noise":
            [9.65565736e+08, 1.64856694e+08, 3.46018289e+08, -6.30177185e+08, -1.81838611e+08, 3.27368113e+08],
        # "With $\mathcal{F}$ noise_4_excitations":
        #     [2.66932321e+08, 8.05425247e+08, 6.50007944e+08, 5.18053354e+07, 2.93691168e+08, -6.65262643e+08]
        # "With $\mathcal{F}$ noise":
        "with_noise":
            [8.31230692e+08, 1.06683247e+09, 1.14319962e+09, - 7.10045638e+08, - 1.61976697e+08, 1.76821149e+09]

    }
    gridspec_kwargs = {
        'top': 0.95,
        'bottom': 0.08,
        # 'left': 0.1,
        # 'right': 0.85,
        'left': 0.22,
        'right': 0.96,
        'hspace': 0
    }
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', gridspec_kw=gridspec_kwargs, figsize=(3.5, 3))
    ax1, ax2 = ax2, ax1

    # Delta_ax = ax1.twinx()
    Delta_ax = ax1
    ax1.xaxis.set_major_formatter(ticker.EngFormatter('s'))

    with timer(f"Creating QutipSpinHam (N={N})"):
        spin_ham = QutipSpinHamiltonian(N)
    psi_0 = tensor([basis(2, 1) for _ in range(N)])
    figure_of_merit_func = _get_f_for_excited_count(NUMBER_OF_EXCITED_WANTED)

    for i, (name, protocol) in enumerate(protocols.items()):
        j = i
        i = 0

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', gridspec_kw=gridspec_kwargs, figsize=(3.5, 3))
        ax1, ax2 = ax2, ax1
        Delta_ax = ax1
        ax1.xaxis.set_major_formatter(ticker.EngFormatter('s'))

        input_ = np.array(protocol)

        # Plot Protocol
        mean_Omega, mean_Delta, Omega_lims, Delta_lims = get_protocol_noise(input_, t_list)
        # Omega_color = f"C{2 * i}"
        # Delta_color = f"C{2 * i + 1}"

        Omega_color = "C0"
        Delta_color = "C3"

        ls = '-' if i == 0 else '--'

        scaling = 1
        ax1.plot(
            t_list, mean_Omega / scaling,
            color=Omega_color,
            lw=3, alpha=0.8, ls=ls
        )
        ax1.fill_between(
            t_list,
            Omega_lims[0] / scaling,
            Omega_lims[1] / scaling,
            color=Omega_color,
            alpha=0.3
        )
        ax1.locator_params(nbins=3, axis='x')

        Delta_ax.plot(t_list, mean_Delta / scaling, color=Delta_color, lw=3, alpha=0.8, ls=ls)
        Delta_ax.fill_between(
            t_list,
            Delta_lims[0] / scaling,
            Delta_lims[1] / scaling,
            color=Delta_color, alpha=0.3
        )

        ax1.yaxis.set_major_formatter(ticker.EngFormatter('Hz'))

        # Calculate F distribution
        def get_fidelity_noise(input_: np.ndarray):
            figure_of_merit_over_times = []
            for i in range(15):
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
                   np.percentile(figure_of_merit_over_times, [15.9, 84.1], axis=stats_axis)

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

        mean_fidelities, fidelities_lims = get_fidelity_noise(input_)
        # Panel 2
        cmap = plt.get_cmap("Set1")
        # cmap = plt.get_cmap("tab10")
        # cmap = plt.get_cmap("Pastel1")
        label = name
        ax2.plot(
            t_list,
            mean_fidelities,
            # label=label,
            lw=3,
            alpha=0.8,
            color=cmap(i / N)
        )
        # ax2.fill_between(
        #     t_list,
        #     fidelities_lims[0] / scaling,
        #     fidelities_lims[1] / scaling,
        #     color=cmap(i / N),
        #     alpha=0.3
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

        plt.savefig(f"bo_paper_plots/N{N}_OPT{NUMBER_OF_EXCITED_WANTED}_{len(geometry.shape)}d_multi_{name}.png",
                    dpi=300)


# print(plt.rcParams["font.size"])
# plt.rcParams["font.size"] = 20
# Plot protocol and fidelity
# gridspec_kwargs = {
#     'top': 0.95,
#     'bottom': 0.08,
#     # 'left': 0.15,
#     # 'right': 0.80,
#     'left': 0.22,
#     'right': 0.96,
#     'hspace': 0
# }
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', gridspec_kw=gridspec_kwargs, figsize=(3.5, 3))
# _plot_protocol_and_fidelity(ax1, ax2, t_list, solve_result,
#                             first=True, last=True)
# plt.savefig(f"bo_paper_plots/N{N}_OPT{NUMBER_OF_EXCITED_WANTED}_{len(geometry.shape)}d.png",
#             dpi=300)
# plt.show()

# Plot populations
# fig, ax1 = plt.subplots(1, 1, figsize=(4.5, 4))
# _plot_atom_excited_population_at_end(ax1, solve_result)
# plt.savefig(
#     f"bo_paper_plots/N{N}_OPT{NUMBER_OF_EXCITED_WANTED}_{len(geometry.shape)}d_"
#     f"populations_T.png",
#     dpi=300
# )
# fig, ax1 = plt.subplots(1, 1)
# _plot_atom_excited_population(ax1, t_list, solve_result)
# plt.savefig(
#     f"bo_paper_plots/N{N}_OPT{NUMBER_OF_EXCITED_WANTED}_{len(geometry.shape)}d_"
#     f"populations.png",
#     dpi=300)
# plt.show()
# plt.show()

# Plot both
# gridspec_kwargs = {
#     'top': 0.95,
#     'bottom': 0.08,
#     'left': 0.1,
#     'right': 0.85,
# }
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all', gridspec_kw=gridspec_kwargs, figsize=(6, 8))
# _plot_protocol_and_fidelity(ax1, ax2, t_list, solve_result,
#                             first=True, last=True)
# _plot_atom_excited_population(ax3, t_list, solve_result)
#
# plt.savefig(f"bo_paper_plots/N{N}_OPT{NUMBER_OF_EXCITED_WANTED}_{len(geometry.shape)}d_full.png",
#             dpi=300)
# plt.show()

protocol_and_fidelity_for_protocols()
