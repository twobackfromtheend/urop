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

plt.rc('font', size=13)

_cmap = plt.cm.get_cmap('plasma_r')(np.linspace(0, 1, 300))
_cmap = _cmap[:-10]
_cmap[:, -1] = np.linspace(0.1, 1, len(_cmap))
COLORMAP = ListedColormap(_cmap)
COLORMAP_CBAR = ListedColormap(_cmap[:, :-1])

NORM = LogNorm(vmin=1e-3, vmax=1, clip=True)


LINE_STYLES = [
    '-',
    '--',
    '-.',
]


def _plot_entropy(ax: Axes, e_qs: EvolvingQubitSystem, i: int):
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
        ls=LINE_STYLES[i],
        color=f'C{i}', linewidth=2,
        alpha=0.9,
        # label=f'{len(e_qs.geometry.shape)}D',
        label=f'{e_qs.N}',
    )

    # ax.set_yticks([0, 1])


def plot():
    plt.rcParams['axes.labelpad'] = 1

    gridspec_kwargs = {
        'nrows': 1,
        'ncols': 1,
        'hspace': 0.1,
        'wspace': 0.1,
        'top': 0.95, 'bottom': 0.23, 'left': 0.28, 'right': 0.95
    }
    gs = GridSpec(**gridspec_kwargs)
    for prefix, func, ylabel in [
        # ('new_entropy_log', _plot_entropy, r"$\log_{10} {\mathcal{S}(\rho_{A})}$"),
        ('new_entropy_log', _plot_entropy, r"$\mathcal{S}(\rho_{A}(t))$"),
    ]:

        fig = plt.figure(figsize=(2.5, 2.5))
        ax = fig.add_subplot(gs[0, 0])

        for i, BO_file in enumerate(BO_FILES):
            e_qs = saver.load(BO_file)
            func(ax, e_qs, i)

            delta = (e_qs.t_list.max() - e_qs.t_list.min()) * 0.01

            t = e_qs.t_list.max()
        #     ax.set_xlim((t * 1e-3, t + delta))
        # ax.set_ylim((0.04, ax.get_ylim()[1]))
        # ax.set_ylim((0.04, ax.get_ylim()[1]))

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([1 * 10 ** (-_x - 6) for _x in range(10)])
        ax.set_yticks([1e-9, 1e-6, 1e-3, 1])
        # ax.set_yticklabels(["$-9$", "$-6$", "$-3$", "$0$"])
        # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:e}'.format(x * 1e6)))
        class XTicker(ticker.LogFormatterMathtext):
            def __call__(self, x, *args, **kwargs):
                return super().__call__(x * 1e6, *args, **kwargs)
        ax.xaxis.set_major_formatter(XTicker())
        # ax.set_xlim((t * 1e-3, t * 1.2))
        ax.set_xlim((1e-9, t * 1.2))
        ax.set_ylim((1e-10, 2))

        # ax.locator_params(nbins=5, axis='x')

        ax.grid(alpha=0.5)
        ax.set_xlabel(r"$t$ [$\upmu$s]")

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
        # ax.legend()

        save_current_fig(f"_plot_10_{prefix}_{'__'.join(BO_FILES)}")


if __name__ == '__main__':
    BO_FILES = [
        # f"12_BO_SHORT_STD_1D_std_",
        # f"12_BO_SHORT_STD_2D_std_",
        # f"12_BO_SHORT_STD_3D_std_",

        f"12_BO_COMPARE_BO_WIDER_1D_alt_",
        f"12_BO_COMPARE_BO_2D_alt_",
        f"12_BO_COMPARE_BO_3D_alt_",


        # "entanglement_entropy_ramp__8_2D_std",
        # "entanglement_entropy_ramp_2_8_2D_std",
        # "entanglement_entropy_ramp__12_2D_std",
        # "entanglement_entropy_ramp__16_2D_std",

        # "entanglement_entropy_ramp__8_2D_alt",
        # "entanglement_entropy_ramp__12_2D_alt",
        # "entanglement_entropy_ramp__16_2D_alt",
    ]

    plot()
