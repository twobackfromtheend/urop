from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.gridspec import GridSpec
import quimb as q

import interaction_constants
from optimised_protocols import saver
from plots_creation.n12_final.utils import save_current_fig
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem


def _plot_N_pc_and_entropy(ax1: Axes, ax2: Axes, e_qs: EvolvingQubitSystem, first: bool):
    # ax1.xaxis.set_major_formatter(ticker.EngFormatter('s'))
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1e6)))
    system_states = e_qs.solved_states
    N_pcs = []
    entropies = []
    for state in system_states:
        sum_powers = np.sum((np.power(np.abs(state), 4)))
        N_pc = 1 / sum_powers
        N_pcs.append(N_pc)
        entropy = q.calc.entropy_subsys(state, [2] * e_qs.N, np.arange(e_qs.N / 2))
        entropies.append(entropy)
    ax1.plot(
        e_qs.solved_t_list, N_pcs,
        color='C0', linewidth=2,
        alpha=0.9
    )
    ax1.locator_params(nbins=3, axis='x')
    ax1.locator_params(nbins=4, axis='y')

    if first:
        ax1.set_ylabel("$N_{PC}$")
        ax2.set_ylabel(r"$\mathcal{S}(\rho_{A})$")
    # else:
        # ax1.get_yaxis().set_ticklabels([])
        # ax2.get_yaxis().set_ticklabels([])
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.plot(
        e_qs.solved_t_list, entropies,
        color='C1', linewidth=2,
        alpha=0.9
    )
    # ax2.locator_params(nbins=3, axis='x')
    # ax2.locator_params(nbins=4, axis='y')
    ax2.set_yticks([0, 1])
    ax2.set_xlabel(r"Time [$\upmu$s]")

    ax1.set_ylim([0, ax1.get_ylim()[1]])
    ax1.grid()
    ax1.set_xlim([0, 1e-6])
    ax2.grid()


def plot_eigenstate_stats(bo_files: List[str], name: str):
    fig = plt.figure(figsize=(8.5, 4.5))
    # gs = fig.add_gridspec(5, 3, wspace=0.3, hspace=0.05, height_ratios=[1, 1, 0.7, 1, 1],
    #                       top=0.95, bottom=0.05, left=0.05, right=0.95)
    gridspec_kwargs = {
        'nrows': 2,
        'ncols': 3,
        'hspace': 0.12,
        'wspace': 0.2,
        'width_ratios': [15, 15, 1],
        'height_ratios': [1, 1],
        'top': 0.88, 'bottom': 0.15, 'left': 0.10, 'right': 0.86
    }
    gs = GridSpec(**gridspec_kwargs)

    # fig, (axs) = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(16, 6), gridspec_kw=gridspec_kwargs)
    for col, BO_file in enumerate(bo_files):
        e_qs = saver.load(BO_file)
        # if ax1 is None:
        ax1 = fig.add_subplot(gs[0, col])
        # else:
        #     ax1 = fig.add_subplot(gs[0, col], sharey=ax1)

        ax2 = fig.add_subplot(gs[1, col], sharex=ax1)

        _plot_N_pc_and_entropy(
            ax1, ax2, e_qs,
            first=col == 0,
        )
        D = len(e_qs.geometry.shape)
        ghz_ket = r"\ghzstdd" if "std" in BO_file else r"\ghzaltd"
        ghz_ket  += "{" + str(D) + "}"
        # ax1.set_title(f"{len(e_qs.geometry.shape)}D, " + r"$\quad \ket{\psi_{\mathrm{target}}} = " + ghz_ket + "$")
        ax1.set_title(f"${ghz_ket}$")

    save_current_fig(name)


if __name__ == '__main__':
    for bo_files, name in [
        ([f"12_BO_COMPARE_BO_WIDER_1D_std_", f"12_BO_COMPARE_BO_WIDER_1D_alt_", ], "eigenstate_stats_1d"),
        ([f"12_BO_COMPARE_BO_2D_std_", f"12_BO_COMPARE_BO_2D_alt_", ], "eigenstate_stats_2d"),
        ([f"12_BO_COMPARE_BO_3D_std_", f"12_BO_COMPARE_BO_3D_alt_", ], "eigenstate_stats_3d"),
    ]:
        plot_eigenstate_stats(bo_files, name)
