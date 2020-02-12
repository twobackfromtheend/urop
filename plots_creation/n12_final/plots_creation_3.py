from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import quimb as q
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from optimised_protocols import saver
from plots_creation.n12_final.utils import save_current_fig
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem

_cmap = plt.cm.get_cmap('plasma_r')(np.linspace(0, 1, 300))
_cmap = _cmap[:-10]
_cmap[:, -1] = np.linspace(0.1, 1, len(_cmap))
COLORMAP = ListedColormap(_cmap)
COLORMAP_CBAR = ListedColormap(_cmap[:, :-1])

NORM = LogNorm(vmin=1e-4, vmax=1, clip=True)

DATA_PATH = Path("inst_eigenstates")


def _save_time_dependent_eigenstates(e_qs: EvolvingQubitSystem, name: str):
    for i in tqdm(range(len(e_qs.Omega))):
        if i % 30 != 0:
            continue
        _Omega = e_qs.Omega[i]
        _Delta = e_qs.Delta[i]
        state = e_qs.solved_states[i]

        hamiltonian = e_qs.get_hamiltonian(_Omega, _Delta)

        dense_hamiltonian = q.qu(hamiltonian, sparse=False)
        instantaneous_eigenenergies, instantaneous_eigenstates = q.eigh(dense_hamiltonian)

        instantaneous_eigenstates = instantaneous_eigenstates.T

        np.savez_compressed(DATA_PATH / f"{name}.{i}.solved_state.npz", state)
        np.savez_compressed(DATA_PATH / f"{name}.{i}.values.npz", instantaneous_eigenenergies)
        np.savez_compressed(DATA_PATH / f"{name}.{i}.states.npz", instantaneous_eigenstates)


def save_time_dependent_energies(BO_FILES):
    for BO_file in BO_FILES:
        e_qs = saver.load(BO_file)
        _save_time_dependent_eigenstates(e_qs, BO_file)


def _load_time_dependent_energies(name: str, i: int):
    with np.load(DATA_PATH / f"{name}.{i}.solved_state.npz") as data:
        state = data[data.files[0]]
    with np.load(DATA_PATH / f"{name}.{i}.values.npz") as data:
        instantaneous_eigenenergies = data[data.files[0]]
    with np.load(DATA_PATH / f"{name}.{i}.states.npz") as data:
        instantaneous_eigenstates = data[data.files[0]]

    return state, instantaneous_eigenenergies, instantaneous_eigenstates
    # return instantaneous_eigenenergies, instantaneous_eigenstates


def _plot_time_dependent_eigenenergies(ax1: Axes, e_qs: EvolvingQubitSystem, BO_file: str, first: bool):
    AXIS_LIMS = []

    ghz_state_tensor = q.qu(e_qs.ghz_state.get_state_tensor(), sparse=False)

    eigenenergies = []
    eigenstate_populations = []
    # ghz_fidelities = []
    plot_t_list = []
    for i in tqdm(range(len(e_qs.Omega))):
        if i % 30 != 0:
            continue
        plot_t_list.append(e_qs.t_list[i])
        try:
            state, instantaneous_eigenenergies, instantaneous_eigenstates = _load_time_dependent_energies(BO_file, i)
        except Exception as e:
            print(f"Could not load saved energies: {e}")
            continue
        _eigenenergies = []
        _eigenstate_populations = []
        _ghz_fidelities = []
        for eigenenergy, eigenstate in zip(instantaneous_eigenenergies, instantaneous_eigenstates):
            eigenenergy = eigenenergy - instantaneous_eigenenergies.min()
            # if eigenenergy > 1e9:
            #     continue
            _eigenenergies.append(eigenenergy)
            eigenstate_population = q.fidelity(state, eigenstate)
            _eigenstate_populations.append(eigenstate_population)
            if eigenstate_population > 0.05:
                AXIS_LIMS.append(eigenenergy)
            _eigenstate_ghz_fidelity = q.fidelity(ghz_state_tensor, eigenstate)
            # _ghz_fidelities.append(_eigenstate_ghz_fidelity)
        eigenenergies.append(_eigenenergies)
        eigenstate_populations.append(_eigenstate_populations)
        # ghz_fidelities.append(_ghz_fidelities)

    eigenenergies = np.array(eigenenergies).T
    eigenstate_populations = np.array(eigenstate_populations).T
    # ghz_fidelities = np.array(ghz_fidelities).T

    xs = []
    ys = []
    cs = []

    for i, _eigenenergies in enumerate(tqdm(eigenenergies)):
        _eigenstate_populations = eigenstate_populations[i]
        # _ghz_fidelities = ghz_fidelities[i]

        x = plot_t_list
        y = _eigenenergies
        c = np.clip(_eigenstate_populations, 1e-4, 1)
        xs.append(x)
        ys.append(y)
        cs.append(c)

        # points = np.array([plot_t_list, _eigenenergies]).T.reshape(-1, 1, 2)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # lc = LineCollection(segments, cmap=COLORMAP, norm=NORM,
        #                     zorder=2 + _eigenstate_populations.max(),
        #                     linewidths=(1,))
        # ax1.add_collection(lc)
        # lc.set_array(np.clip(_eigenstate_populations, 1e-4, 1))

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    cs = np.concatenate(cs)
    order = np.argsort(cs)
    # print(len(order))
    ax1.scatter(
        x=xs[order],
        y=ys[order],
        c=cs[order],
        cmap=COLORMAP, norm=NORM,
        s=4,
        edgecolors='none',
    )

    delta = (e_qs.t_list.max() - e_qs.t_list.min()) * 0.01
    ax1.set_xlim((e_qs.t_list.min() - delta, e_qs.t_list.max() + delta))

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1e6)))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1e9)))

    if first:
        ax1.set_ylabel('[GHz]')
    ax1.locator_params(nbins=4, axis='y')
    ax1.locator_params(nbins=5, axis='x')

    min_energy = min(AXIS_LIMS)
    max_energy = max(AXIS_LIMS)
    range_energy = max_energy - min_energy

    buffer = range_energy * 0.1
    ax1.set_ylim((min_energy - buffer, max_energy + buffer))

    ax1.grid(alpha=0.5)
    ax1.set_xlabel(r"[$\upmu$s]")


def plot_time_dependent_eigenenergies():
    gridspec_kwargs = {
        'nrows': 1,
        'ncols': 2,
        'hspace': 0.1,
        'wspace': 0.1,
        'width_ratios': [20, 1],
        'top': 0.95, 'bottom': 0.2, 'left': 0.15, 'right': 0.8
    }
    gs = GridSpec(**gridspec_kwargs)

    # fig, (axs) = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(16, 6), gridspec_kw=gridspec_kwargs)
    ax1 = ax2 = None
    for col, BO_file in enumerate(BO_FILES):
        fig = plt.figure(figsize=(5.5, 3))

        # e_qs = saver.load(BO_file)
        e_qs = saver.load(BO_file, solve=False)

        ax1 = fig.add_subplot(gs[0, 0])

        _plot_time_dependent_eigenenergies(
            ax1, e_qs,
            BO_file,
            first=True,
        )

        cax = fig.add_subplot(gs[0, 1])
        mappable = ScalarMappable(norm=NORM, cmap=COLORMAP_CBAR)

        cbar = plt.colorbar(mappable, cax=cax, ticks=[1e-4,1e-3, 1e-2, 1e-1, 1], extend='min')
        # # cbar.ax.set_yticklabels(['$< 0.001$', '$0.01$', '$0.1$', '$1$'])
        cbar.ax.set_yticklabels(['$< 10^{-4}$', '$10^{-3}$','$10^{-2}$', '$10^{-1}$', '$1$'])

        # cbar = plt.colorbar(mappable, cax=cax, ticks=[1e-3, 1e-2, 1e-1, 1], extend='min')
        # # cbar.ax.set_yticklabels(['$< 0.001$', '$0.01$', '$0.1$', '$1$'])
        # cbar.ax.set_yticklabels(['$< 10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])
        cbar.ax.set_ylabel(r"Eigenstate population")

        save_current_fig(f"inst_states_{BO_file}")


if __name__ == '__main__':
    BO_FILES = [
        f"12_BO_COMPARE_BO_3D_std_",
        f"12_BO_COMPARE_BO_3D_alt_",
        f"12_BO_COMPARE_BO_2D_std_",
        f"12_BO_COMPARE_BO_2D_alt_",
        f"12_BO_COMPARE_BO_1D_std_",
        f"12_BO_COMPARE_BO_1D_alt_",
    ]
    # save_time_dependent_energies(BO_FILES)
    plot_time_dependent_eigenenergies()
