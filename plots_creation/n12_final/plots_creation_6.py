from collections import defaultdict

import numpy as np
import quimb as q
from matplotlib import pyplot as plt, ticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from optimised_protocols import saver
from plots_creation.n12_final.operators import get_total_M_opt, get_total_kinks_opt
from plots_creation.n12_final.plots_creation_3 import _load_time_dependent_energies
from plots_creation.n12_final.utils import save_current_fig

_cmap = plt.cm.get_cmap('plasma_r')(np.linspace(0, 1, 300))
_cmap = _cmap[:-10]
_cmap[:, -1] = np.linspace(0.3, 1, len(_cmap))
COLORMAP = ListedColormap(_cmap)
COLORMAP_CBAR = ListedColormap(_cmap[:, :-1])

NORM = Normalize(vmin=0, vmax=1, clip=True)


def plot_kinks():
    N = 12

    total_kinks_opt = get_total_kinks_opt(N)

    for col, BO_file in enumerate(BO_FILES):
        e_qs = saver.load(BO_file, solve=False)

        eigenenergies = []
        eigenstate_populations = []
        ghz_fidelities = []
        plot_t_list = []

        inst_state_stats = []
        for i in tqdm(range(len(e_qs.Omega))):
            if i % 30 != 0:
                continue
            _t = e_qs.t_list[i]
            plot_t_list.append(_t)
            try:
                state, instantaneous_eigenenergies, instantaneous_eigenstates = _load_time_dependent_energies(BO_file,
                                                                                                              i)
            except Exception as e:
                print(f"Could not load saved energies: {e}")
                continue

            _eigenenergies = []
            _eigenstate_populations = []
            _ghz_fidelities = []
            for i, (eigenenergy, eigenstate) in enumerate(zip(instantaneous_eigenenergies, instantaneous_eigenstates)):
                eigenenergy = eigenenergy - instantaneous_eigenenergies.min()
                # if eigenenergy > 1e9:
                #     continue
                _eigenenergies.append(eigenenergy)
                eigenstate_population = q.fidelity(state, eigenstate)
                _eigenstate_populations.append(eigenstate_population)
                # _eigenstate_ghz_fidelity = q.fidelity(ghz_state_tensor, eigenstate)
                # _ghz_fidelities.append(_eigenstate_ghz_fidelity)

                if eigenstate_population > 0.05:
                    expected_kinks = q.expec(total_kinks_opt, eigenstate).real
                    inst_state_stat = (i, _t, eigenstate_population, expected_kinks)
                    print(inst_state_stat)
                    inst_state_stats.append(inst_state_stat)

            eigenenergies.append(_eigenenergies)
            eigenstate_populations.append(_eigenstate_populations)
            ghz_fidelities.append(_ghz_fidelities)

        gridspec_kwargs = {
            'nrows': 1,
            'ncols': 2,
            'wspace': 0.1,
            'width_ratios': [15, 1],
            'top': 0.96, 'bottom': 0.18, 'left': 0.2, 'right': 0.8
        }
        gs = GridSpec(**gridspec_kwargs)
        fig = plt.figure(figsize=(5, 3.5))
        ax1 = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[:, 1])

        plot_data = {
            't': defaultdict(list),
            'm': defaultdict(list),
            'n': defaultdict(list),
            'c': defaultdict(list)
        }
        i_s = set()
        for i, _t, eigenstate_population, expected_kinks in inst_state_stats:
            i_s.add(i)
            plot_data['t'][i].append(_t)
            plot_data['n'][i].append(expected_kinks)
            plot_data['c'][i].append(eigenstate_population)

        for i in i_s:
            xs = plot_data['t'][i]
            ys = plot_data['n'][i]
            cs = plot_data['c'][i]
            ax1.scatter(x=xs, y=ys, c=cs, s=5, cmap=COLORMAP, norm=NORM, edgecolors='none')

            # points = np.array([xs, ys]).T.reshape(-1, 1, 2)
            # segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # lc = LineCollection(segments, cmap=COLORMAP, norm=NORM, linewidths=(3,))
            # ax2.add_collection(lc)
            # lc.set_array(np.clip(cs, 1e-4, 1))

        ax1.set_ylabel(r"Kinks $\mathcal{N}$")

        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1e6)))

        ax1.set_xlabel(r"[$\upmu$s]")
        # plt.setp(ax1.get_xticklabels(), visible=False)

        ax1.grid()

        mappable = ScalarMappable(norm=NORM, cmap=COLORMAP_CBAR)
        cbar = plt.colorbar(mappable, cax=cax)
        # cbar = plt.colorbar(mappable, cax=cax, ticks=[1e-3, 1e-2, 1e-1, 1], extend='min')
        # cbar.ax.set_yticklabels(['$< 10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])
        cbar.ax.set_ylabel(r"Eigenstate population")

        ax1.set_xlim((-0.1e-6, 1.1e-6))
        ax1.set_ylim((0, max(max(kinks) for kinks in plot_data['n'].values())))
        # plt.tight_layout()
        # plt.show()
        save_current_fig(f"total_kinks_{BO_file}")


if __name__ == '__main__':
    BO_FILES = [
        f"12_BO_COMPARE_BO_3D_std_",
        f"12_BO_COMPARE_BO_3D_alt_",
        f"12_BO_COMPARE_BO_2D_std_",
        f"12_BO_COMPARE_BO_2D_alt_",
        f"12_BO_COMPARE_BO_1D_std_",
        f"12_BO_COMPARE_BO_1D_alt_",
    ]
    plot_kinks()
