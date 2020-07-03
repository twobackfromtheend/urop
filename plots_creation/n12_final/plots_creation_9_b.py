from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import quimb as q

from optimised_protocols import saver
from plots_creation.n12_final.operators import get_total_M_opt
from plots_creation.n12_final.plots_creation_3 import _load_time_dependent_energies
from plots_creation.n12_final.utils import save_current_fig
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem


_cmap = plt.cm.get_cmap('plasma_r')(np.linspace(0, 1, 300))
_cmap = _cmap[:-10]
_cmap[:, -1] = np.linspace(0.1, 1, len(_cmap))
COLORMAP = ListedColormap(_cmap)
COLORMAP_CBAR = ListedColormap(_cmap[:, :-1])

NORM = LogNorm(vmin=1e-3, vmax=1, clip=True)



def _plot_entropy(ax: Axes, e_qs: EvolvingQubitSystem, BO_file: str):
    system_states = e_qs.solved_states
    # N_pcs = []
    entropies = []
    for state in tqdm(system_states):
        # sum_powers = np.sum((np.power(np.abs(state), 4)))
        # N_pc = 1 / sum_powers
        # N_pcs.append(N_pc)
        entropy = q.calc.entropy_subsys(state, [2] * e_qs.N, np.arange(e_qs.N / 2))
        entropies.append(entropy)

    ax.plot(
        e_qs.solved_t_list, entropies,
        color='C1', linewidth=2,
        alpha=0.9
    )

    ax.set_yticks([0, 1])


def plot():
    plt.rcParams['axes.labelpad'] = 1

    gridspec_kwargs = {
        'nrows': 1,
        'ncols': 1,
        'hspace': 0.1,
        'wspace': 0.1,
        'top': 0.95, 'bottom': 0.23, 'left': 0.2, 'right': 0.95
    }
    gs = GridSpec(**gridspec_kwargs)
    for BO_file in BO_FILES:
        e_qs = saver.load(BO_file)

        for prefix, func, ylabel in [
            ('new_entropy', _plot_entropy, r"$\mathcal{S}(\rho_{A})$"),
        ]:
            fig = plt.figure(figsize=(3.5, 2.5))
            ax = fig.add_subplot(gs[0, 0])

            func(ax, e_qs, BO_file)

            delta = (e_qs.t_list.max() - e_qs.t_list.min()) * 0.01
            ax.set_xlim((e_qs.t_list.min() - delta, e_qs.t_list.max() + delta))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1e6)))

            ax.locator_params(nbins=5, axis='x')

            ax.grid(alpha=0.5)
            ax.set_xlabel(r"Time [$\upmu$s]")

            ax.text(
                0.02, 0.59,
                ylabel,
                horizontalalignment='left',
                verticalalignment='center',
                rotation='vertical',
                # textcoords='figure points',
                transform=fig.transFigure,
                # transform=ax.transAxes
            )

            save_current_fig(f"_plot_9_{prefix}_{BO_file}")


if __name__ == '__main__':
    BO_FILES = [
        "entanglement_entropy_ramp__8_2D_std",
        "entanglement_entropy_ramp__12_2D_std",
        "entanglement_entropy_ramp__16_2D_std",
        "entanglement_entropy_ramp__8_2D_alt",
        "entanglement_entropy_ramp__12_2D_alt",
        "entanglement_entropy_ramp__16_2D_alt",
    ]

    plot()
