from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

from optimised_protocols import saver
from plots_creation.n12_final.utils import save_current_fig


def plot_fermion_eigenenergy(bo_files: List[Tuple[str, float]]):
    for BO_file, V_0 in bo_files:
        # fig = plt.figure(figsize=(4.5, 3))
        fig = plt.figure(figsize=(4, 3))

        e_qs = saver.load(BO_file, solve=False)

        N = e_qs.N
        Omega = e_qs.Omega

        n = N - 1
        # n = N

        e_k = {}
        for _i in range(1, N):
        # for i in range(0, N):
            i = _i + 1
            ka = i * np.pi / N
            e_k_i = 2 * V_0 * np.sqrt((Omega / V_0 - np.cos(ka)) ** 2 + np.sin(ka) ** 2)
            e_k_i = np.hstack([e_k_i, e_k_i[-1]])
            e_k[i] = e_k_i

        # from_list = LinearSegmentedColormap.from_list
        # colors = plt.cm.Set3(range(0, n))
        # cmap = from_list(None, colors, n)

        for i, e_k_i in e_k.items():
            # plt.plot(e_qs.t_list, e_k_i, label=i, c=colors[i])
            plt.plot(e_qs.t_list, e_k_i, label=i, c='C0', alpha=0.8)
        # plt.legend()
        plt.xlabel(r"Time [$\upmu$s]")
        plt.ylabel('$\epsilon_k$ [GHz]')

        # mappable = ScalarMappable(cmap=cmap)
        # cbar = plt.colorbar(mappable,
        #                     ticks=np.linspace(0, 1, n + 1) + 1 / (2 * (n + 1)),
        #                     label='$k$')
        # cbar.ax.set_yticklabels(list(e_k.keys()))

        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1e6)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1e9)))
        ax.locator_params(nbins=5, axis='x')
        ax.locator_params(nbins=4, axis='y')

        plt.tight_layout(pad=0.3)
        # plt.show()
        save_current_fig(f"fermion_energy_{BO_file}")


if __name__ == '__main__':
    BO_FILES = [
        (f"12_BO_COMPARE_BO_WIDER_1D_std_", 1.271e9),
        (f"12_BO_COMPARE_BO_WIDER_1D_alt_", 1.802e8),
        (f"12_BO_COMPARE_BO_2D_std_", 2.137e9),
        (f"12_BO_COMPARE_BO_2D_alt_", 1.898e8),
        (f"12_BO_COMPARE_BO_3D_std_", 2.639e9),
        (f"12_BO_COMPARE_BO_3D_alt_", 3.221e8),
    ]
    plot_fermion_eigenenergy(BO_FILES)
