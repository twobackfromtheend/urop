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

# COLORMAP = 'viridis_r'
_cmap = plt.cm.get_cmap('Blues')(np.linspace(0, 1, 100) ** 0.5)
# _cmap = plt.cm.get_cmap('gist_yarg')(np.linspace(0, 1, 100) ** 0.7)
_cmap = _cmap[:-10]
COLORMAP = ListedColormap(_cmap)
COLORMAP_CBAR = ListedColormap(_cmap[:, :-1])
NORM = LogNorm(vmin=1e-4, vmax=1, clip=True)


# _cmap = plt.cm.get_cmap('plasma_r')(np.linspace(0, 1, 300))
# _cmap = _cmap[:-10]
# # _cmap[:, -1] = np.linspace(0.3, 1, len(_cmap))
# COLORMAP = ListedColormap(_cmap)
# COLORMAP_CBAR = ListedColormap(_cmap[:, :-1])
# NORM = LogNorm(vmin=1e-3, vmax=1, clip=True)


def _plot_N_pc_and_populations(ax1: Axes, ax2: Axes, e_qs: EvolvingQubitSystem, first: bool):
    # ax1.xaxis.set_major_formatter(ticker.EngFormatter('s'))
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1e6)))
    system_states = e_qs.solved_states
    N_pcs = []
    for state in system_states:
        sum_powers = np.sum((np.power(np.abs(state), 4)))
        N_pc = 1 / sum_powers
        N_pcs.append(N_pc)
    ax1.plot(
        e_qs.solved_t_list, N_pcs,
        color='C0', linewidth=.5,
        alpha=0.9
    )
    ax1.locator_params(nbins=3, axis='x')
    ax1.locator_params(nbins=4, axis='y')

    if first:
        ax1.set_ylabel("$N_{PC}$")
        ax2.set_ylabel(r"Eigenstates")
    else:
        # ax1.get_yaxis().set_ticklabels([])
        ax2.get_yaxis().set_ticklabels([])
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Eigenstate plot
    cs = []
    for i, state in enumerate(system_states):
        state_abs = np.abs(state)
        sum_powers = np.sum((np.power(state_abs, 4)))
        N_pc = 1 / sum_powers
        N_pcs.append(N_pc)

        if i % 5 == 0:
            populations = np.power(state_abs, 2)
            populations[populations < 1e-16] = 1e-16
            # populations = np.sort(populations.flatten())
            populations = populations.flatten()

            cs.append(populations)

            # for limit in limits:
            #     count_above_limit = np.sum(populations >= limit)
            #     counts[limit].append(count_above_limit)

    eigenenergies, eigenstates = q.eigh(q.qu(e_qs.get_hamiltonian(0, -1), sparse=False))

    eigenenergies_inds = eigenenergies.argsort()
    # print(eigenenergies.shape)
    eigenstates = eigenstates.T
    product_basis_is = []
    for i, eigenstate in enumerate(eigenstates[eigenenergies_inds]):
        product_basis_index = eigenstate.argmax()
        product_basis_is.append(product_basis_index)

    product_basis_is = np.array(product_basis_is)

    cs = np.array(cs).transpose()
    cs = cs[product_basis_is]

    number_of_states = 2 ** e_qs.N
    ax2.imshow(
        cs,
        aspect='auto',
        cmap=COLORMAP, norm=NORM,
        origin='lower',
        extent=(0, e_qs.solved_t_list[-1], 0, number_of_states)
    )
    # for spine in ax2.spines.values():
    #     spine.set_visible(False)

    ax2.set_xlabel(r"Time [$\upmu$s]")
    ax2.set_ylim((0, number_of_states))

    # ax2.set_ylim((number_of_states - 50, number_of_states))
    # ax2.set_ylim((0, 50))

    # ax1.set_ylim([0, ax1.get_ylim()[1]])
    ax1.set_ylim([0, ax1.get_ylim()[1]])
    ax1.grid()
    ax1.set_xlim([0, 1e-6])


def plot_N_pc(bo_files: List[str], names: str):
    fig = plt.figure(figsize=(8.5, 4.5))
    # gs = fig.add_gridspec(5, 3, wspace=0.3, hspace=0.05, height_ratios=[1, 1, 0.7, 1, 1],
    #                       top=0.95, bottom=0.05, left=0.05, right=0.95)
    gridspec_kwargs = {
        'nrows': 2,
        'ncols': 3,
        'hspace': 0.1,
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

        _plot_N_pc_and_populations(
            ax1, ax2, e_qs,
            first=col == 0,
        )
        D = len(e_qs.geometry.shape)
        ghz_ket = r"\ghzstdd" if "std" in BO_file else r"\ghzaltd"
        ghz_ket  += "{" + str(D) + "}"
        # ax1.set_title(f"{len(e_qs.geometry.shape)}D, " + r"$\quad \ket{\psi_{\mathrm{target}}} = " + ghz_ket + "$")
        ax1.set_title(f"${ghz_ket}$")

    cax = fig.add_subplot(gs[1, 2])
    mappable = ScalarMappable(norm=NORM, cmap=COLORMAP_CBAR)

    # cbar = plt.colorbar(mappable, cax=cax, ticks=[1e-3, 1e-2, 1e-1, 1], extend='min')
    # cbar.ax.set_yticklabels(['$< 10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])

    cbar = plt.colorbar(mappable, cax=cax, ticks=[1e-4, 1e-3, 1e-2, 1e-1, 1], extend='min')
    # cbar.ax.set_yticklabels(['$< 0.0001$', '$0.001$', '$0.01$', '$0.1$', '$1$'])
    cbar.ax.set_yticklabels(['$< 10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])
    cbar.ax.set_ylabel(r"Eigenstate population")
    # plt.show()
    save_current_fig(name)
    # save_current_fig(name + "_zoomed_high")
    # save_current_fig(name + "_zoomed_low")


if __name__ == '__main__':
    for bo_files, name in [
        ([f"12_BO_COMPARE_BO_WIDER_1D_std_", f"12_BO_COMPARE_BO_WIDER_1D_alt_", ], "npc_1d"),
        # ([f"12_BO_COMPARE_BO_1D_std_", f"12_BO_COMPARE_BO_1D_alt_", ], "npc_1d"),
        # ([f"12_BO_COMPARE_BO_2D_std_", f"12_BO_COMPARE_BO_2D_alt_", ], "npc_2d"),
        # ([f"12_BO_COMPARE_BO_3D_std_", f"12_BO_COMPARE_BO_3D_alt_", ], "npc_3d"),
    ]:
        plot_N_pc(bo_files, name)
