import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import quimb as q

import interaction_constants
from optimised_protocols import saver
from plots_creation.utils import save_current_fig
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem

N = 8
LATTICE_SPACING = 1.5e-6

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

COLORMAP = 'viridis_r'
NORM = LogNorm(vmin=1e-3, vmax=1, clip=True)


def _plot_N_pc_and_populations(ax1: Axes, ax2: Axes, e_qs: EvolvingQubitSystem, first: bool):
    ax1.xaxis.set_major_formatter(ticker.EngFormatter('s'))
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
        ax1.get_yaxis().set_ticklabels([])
        ax2.get_yaxis().set_ticklabels([])

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

    ax2.imshow(
        cs,
        aspect='auto',
        cmap=plt.cm.get_cmap(COLORMAP), norm=NORM,
        origin='lower',
        extent=(0, e_qs.solved_t_list[-1], 0, 256)
    )
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax2.set_ylim((0, 256))
    # ax1.set_ylim([0, ax1.get_ylim()[1]])
    ax1.set_ylim([0, 200])
    ax1.grid()
    ax1.set_xlim([0, 1e-6])


def plot_N_pc():
    fig = plt.figure(figsize=(8, 6))
    # gs = fig.add_gridspec(5, 3, wspace=0.3, hspace=0.05, height_ratios=[1, 1, 0.7, 1, 1],
    #                       top=0.95, bottom=0.05, left=0.05, right=0.95)
    gridspec_kwargs = {
        'nrows': 2,
        'ncols': 3,
        'hspace': 0.1,
        'wspace': 0.2,
        'width_ratios': [15, 15, 1],
        'height_ratios': [1, 2],
        'top': 0.95, 'bottom': 0.05, 'left': 0.08, 'right': 0.90
    }
    gs = GridSpec(**gridspec_kwargs)

    # fig, (axs) = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(16, 6), gridspec_kw=gridspec_kwargs)
    ax1 = ax2 = None
    for col, BO_file in enumerate([
        f"BO_COMPARE_BO_3D_std_8",
        f"BO_COMPARE_BO_3D_alt_8"
    ]):
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
        ghz_ket = r"\ghzstd" if "std" in BO_file else r"\ghzalt"
        ax1.set_title(f"3D,  " + r"$\qquad \ket{\psi_{\mathrm{target}}} = " + ghz_ket + "$")

    cax = fig.add_subplot(gs[1, 2])
    mappable = ScalarMappable(norm=NORM, cmap=COLORMAP)
    cbar = plt.colorbar(mappable, cax=cax, ticks=[1e-3, 1e-2, 1e-1, 1], extend='min')
    cbar.ax.set_yticklabels(['$< 0.001$', '$0.01$', '$0.1$', '$1$'])
    cbar.ax.set_ylabel(r"Eigenstate population")

    save_current_fig("npc")


if __name__ == '__main__':
    plot_N_pc()
