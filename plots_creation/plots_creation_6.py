import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import quimb as q

from optimised_protocols import saver
from plots_creation.utils import save_current_fig
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem

COLORMAP = 'viridis_r'
NORM = LogNorm(vmin=1e-3, vmax=1, clip=True)


def _plot_time_dependent_eigenenergies(ax1: Axes, e_qs: EvolvingQubitSystem, first: bool):
    ghz_state_tensor = q.qu(e_qs.ghz_state.get_state_tensor(), sparse=False)

    eigenenergies = []
    eigenstate_populations = []
    ghz_fidelities = []
    plot_t_list = []
    for i in range(len(e_qs.Omega)):
        if i % 10 != 0:
            continue
        plot_t_list.append(e_qs.t_list[i])
        _Omega = e_qs.Omega[i]
        _Delta = e_qs.Delta[i]
        state = e_qs.solved_states[i]

        hamiltonian = e_qs.get_hamiltonian(_Omega, _Delta)

        dense_hamiltonian = q.qu(hamiltonian, sparse=False)
        instantaneous_eigenenergies, instantaneous_eigenstates = q.eigh(dense_hamiltonian)
        instantaneous_eigenstates = instantaneous_eigenstates.T

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
            _eigenstate_ghz_fidelity = q.fidelity(ghz_state_tensor, eigenstate)
            _ghz_fidelities.append(_eigenstate_ghz_fidelity)
        eigenenergies.append(_eigenenergies)
        eigenstate_populations.append(_eigenstate_populations)
        ghz_fidelities.append(_ghz_fidelities)

    eigenenergies = np.array(eigenenergies).T
    eigenstate_populations = np.array(eigenstate_populations).T
    ghz_fidelities = np.array(ghz_fidelities).T

    for i, _eigenenergies in enumerate(eigenenergies):
        _eigenstate_populations = eigenstate_populations[i]
        _ghz_fidelities = ghz_fidelities[i]
        # color = cmap(np.log10(np.clip(_eigenstate_populations, 1e-4, 1)))
        # plt.scatter(plot_t_list, _eigenenergies,
        #             s=3,
        #             c=np.clip(_eigenstate_populations, 1e-4, 1),
        #             # c=color,
        #             # c=np.clip(_ghz_fidelities, 1e-4, 1),
        #             cmap=plt.cm.get_cmap(COLORMAP), norm=NORM,
        #             # c='k',
        #             edgecolors='none',
        #             alpha=0.4)

        points = np.array([plot_t_list, _eigenenergies]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plt.get_cmap(COLORMAP), norm=NORM, alpha=0.5)
        ax1.add_collection(lc)
        lc.set_array(np.clip(_eigenstate_populations, 1e-4, 1))

    delta = (e_qs.t_list.max() - e_qs.t_list.min()) * 0.01
    ax1.set_xlim((e_qs.t_list.min() - delta, e_qs.t_list.max() + delta))

    # ax1.xaxis.set_major_formatter(ticker.EngFormatter('s'))
    # ax1.yaxis.set_major_formatter(ticker.EngFormatter('Hz'))
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1e6)))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1e9)))


    if first:
        ax1.set_ylabel('Eigenenergies')
    ax1.locator_params(nbins=4, axis='y')
    ax1.locator_params(nbins=5, axis='x')

    # ax1.set_facecolor('k')
    ax1.grid(alpha=0.5)
    ax1.set_xlabel(r"Time [$\upmu$s]")


def plot_time_dependent_eigenenergies():
    fig = plt.figure(figsize=(9, 3))
    # gs = fig.add_gridspec(5, 3, wspace=0.3, hspace=0.05, height_ratios=[1, 1, 0.7, 1, 1],
    #                       top=0.95, bottom=0.05, left=0.05, right=0.95)
    gridspec_kwargs = {
        'nrows': 1,
        'ncols': 4,
        'hspace': 0.1,
        'wspace': 0.1,
        'width_ratios': [15, 1, 15, 1],
        'top': 0.84, 'bottom': 0.2, 'left': 0.07, 'right': 0.87
    }
    gs = GridSpec(**gridspec_kwargs)

    # fig, (axs) = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(16, 6), gridspec_kw=gridspec_kwargs)
    ax1 = ax2 = None
    for col, BO_file in enumerate([
        f"BO_COMPARE_BO_3D_std_8",
        f"BO_COMPARE_BO_3D_alt_8"
    ]):
        e_qs = saver.load(BO_file)

        ax1 = fig.add_subplot(gs[0, col * 2])

        _plot_time_dependent_eigenenergies(
            ax1, e_qs,
            first=col == 0,
        )
        ghz_ket = r"\ghzstd" if "std" in BO_file else r"\ghzalt"
        ax1.set_title(f"3D, " + r"$\quad \ket{\psi_{\mathrm{target}}} = " + ghz_ket + "$")

    cax = fig.add_subplot(gs[0, 3])
    mappable = ScalarMappable(norm=NORM, cmap=COLORMAP)
    cbar = plt.colorbar(mappable, cax=cax, ticks=[1e-3, 1e-2, 1e-1, 1], extend='min')
    cbar.ax.set_yticklabels(['$< 0.001$', '$0.01$', '$0.1$', '$1$'])
    cbar.ax.set_ylabel(r"Eigenstate population")

    save_current_fig("inst_states")


if __name__ == '__main__':
    plot_time_dependent_eigenenergies()
