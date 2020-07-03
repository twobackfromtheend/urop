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


def _plot_inst_states(ax: Axes, e_qs: EvolvingQubitSystem, BO_file: str):
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
            state, instantaneous_eigenenergies, instantaneous_eigenstates = _load_time_dependent_energies(BO_file,
                                                                                                          i)
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

    for i, _eigenenergies in enumerate(eigenenergies):
        _eigenstate_populations = eigenstate_populations[i]
        # _ghz_fidelities = ghz_fidelities[i]

        x = plot_t_list
        y = _eigenenergies
        # c = np.clip(_eigenstate_populations, 1e-4, 1)
        c = np.clip(_eigenstate_populations, 1e-3, 1)
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
    ax.scatter(
        x=xs[order],
        y=ys[order],
        c=cs[order],
        cmap=COLORMAP, norm=NORM,
        s=4,
        edgecolors='none',
    )

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1e9)))

    # ax.set_ylabel('$E_N$ [GHz]')
    ax.locator_params(nbins=4, axis='y')

    min_energy = min(AXIS_LIMS)
    max_energy = max(AXIS_LIMS)
    range_energy = max_energy - min_energy

    buffer = range_energy * 0.1
    ax.set_ylim((min_energy - buffer, max_energy + buffer))


def _plot_inst_states_components(ax: Axes, e_qs: EvolvingQubitSystem, BO_file: str):
    ghz_state_tensor = q.qu(e_qs.ghz_state.get_state_tensor(), sparse=False)

    eigenstates_with_populations = []
    plot_t_list = []
    for i in tqdm(range(len(e_qs.Omega))):
        if i % 30 != 0:
            continue
        plot_t_list.append(e_qs.t_list[i])
        try:
            state, instantaneous_eigenenergies, instantaneous_eigenstates = _load_time_dependent_energies(BO_file,
                                                                                                          i)
        except Exception as e:
            print(f"Could not load saved energies: {e}")
            continue
        _eigenstates_with_populations = []
        for eigenenergy, eigenstate in zip(instantaneous_eigenenergies, instantaneous_eigenstates):
            # eigenenergy = eigenenergy - instantaneous_eigenenergies.min()
            eigenstate_population = q.fidelity(state, eigenstate)
            if eigenstate_population > 0.1:
                _eigenstates_with_populations.append((eigenstate, eigenstate_population))
        eigenstates_with_populations.append(_eigenstates_with_populations)

    print(eigenstates_with_populations)


    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1e9)))

    # ax.set_ylabel('$E_N$ [GHz]')
    ax.locator_params(nbins=4, axis='y')



def _plot_magnetisation(ax: Axes, e_qs: EvolvingQubitSystem, BO_file: str):
    N = 12

    total_M_opt = get_total_M_opt(N)

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
            state, instantaneous_eigenenergies, instantaneous_eigenstates = _load_time_dependent_energies(
                BO_file,
                i
            )
        except Exception as e:
            print(f"Could not load saved energies: {e}")
            continue

        _eigenenergies = []
        _eigenstate_populations = []
        _ghz_fidelities = []
        for i, (eigenenergy, eigenstate) in enumerate(
                zip(instantaneous_eigenenergies, instantaneous_eigenstates)):
            eigenenergy = eigenenergy - instantaneous_eigenenergies.min()
            # if eigenenergy > 1e9:
            #     continue
            _eigenenergies.append(eigenenergy)
            eigenstate_population = q.fidelity(state, eigenstate)
            _eigenstate_populations.append(eigenstate_population)
            # if eigenstate_population >= 0.001:
            if True:
                expected_M = q.expec(total_M_opt, eigenstate)
                inst_state_stat = (i, _t, eigenstate_population, expected_M)
                # print(inst_state_stat)
                inst_state_stats.append(inst_state_stat)

        eigenenergies.append(_eigenenergies)
        eigenstate_populations.append(_eigenstate_populations)
        ghz_fidelities.append(_ghz_fidelities)

    # plot_data = {
    #     't': defaultdict(list),
    #     'm': defaultdict(list),
    #     'n': defaultdict(list),
    #     'c': defaultdict(list)
    # }
    # i_s = set()
    # for i, _t, eigenstate_population, expected_M in inst_state_stats:
    #     i_s.add(i)
    #     plot_data['t'][i].append(_t)
    #     plot_data['m'][i].append(expected_M)
    #     plot_data['c'][i].append(eigenstate_population)
    # for i in i_s:
    #     xs = np.array(plot_data['t'][i])
    #     ys = np.array(plot_data['m'][i])
    #     cs = np.array(plot_data['c'][i])
    #     # plt.plot(xs, ys, c=cs, cmap=COLORMAP)
    #
    #     ax.scatter(x=xs, y=ys, c=cs, s=5, cmap=COLORMAP, norm=NORM, edgecolors='none')
    #     # points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    #     # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #     # lc = LineCollection(segments, cmap=COLORMAP, norm=NORM, linewidths=(3,))
    #     # ax1.add_collection(lc)
    #     # lc.set_array(np.clip(cs, 1e-4, 1))

    xs = []
    ys = []
    cs = []
    for i, _t, eigenstate_population, expected_M in inst_state_stats:
        xs.append(_t)
        ys.append(expected_M)
        cs.append(eigenstate_population)
    xs = np.array(xs)
    ys = np.array(ys)
    cs = np.array(cs)
    sorting = np.argsort(cs)
    ax.scatter(x=xs[sorting], y=ys[sorting], c=cs[sorting], s=5, cmap=COLORMAP, norm=NORM, edgecolors='none')

    # ax.set_ylabel("$\mathcal{M}$")

    # mappable = ScalarMappable(norm=NORM, cmap=COLORMAP_CBAR)
    # cbar = plt.colorbar(mappable, cax=cax)
    # cbar = plt.colorbar(mappable, cax=cax, ticks=[1e-3, 1e-2, 1e-1, 1])
    # cbar.ax.set_yticklabels(['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])
    # cbar = plt.colorbar(mappable, cax=cax, ticks=[1e-3, 1e-2, 1e-1, 1], extend='min')
    # cbar.ax.set_yticklabels(['$< 10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])
    # cbar.ax.set_ylabel(r"Eigenstate population")

    ax.set_yticks([-N, 0, N])
    ax.set_ylim((-N - 1, N + 1))


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


def _plot_population(ax: Axes, e_qs: EvolvingQubitSystem, BO_file: str):
    system_states = e_qs.solved_states
    # Eigenstate plot
    cs = []
    for i, state in enumerate(system_states):
        if i % 5 == 0:
            state_abs = np.abs(state)
            populations = np.power(state_abs, 2)
            populations[populations < 1e-16] = 1e-16
            # populations = np.sort(populations.flatten())
            populations = populations.flatten()

            cs.append(populations)

    eigenenergies, eigenstates = q.eigh(q.qu(e_qs.get_hamiltonian(0, -1), sparse=False))

    eigenenergies_inds = eigenenergies.argsort()
    # print(eigenenergies.shape)
    eigenstates = eigenstates.T
    product_basis_indexes = []
    for i, eigenstate in enumerate(eigenstates[eigenenergies_inds]):
        product_basis_index = eigenstate.argmax()
        product_basis_indexes.append(product_basis_index)

    product_basis_indexes = np.array(product_basis_indexes)

    cs = np.array(cs).transpose()
    cs = cs[product_basis_indexes]

    number_of_states = 2 ** e_qs.N
    ax.imshow(
        cs,
        aspect='auto',
        cmap=COLORMAP, norm=NORM,
        origin='lower',
        extent=(0, e_qs.solved_t_list[-1], 0, number_of_states)
    )
    ax.set_ylim((0 - 1, number_of_states + 1))
    N = e_qs.N
    ax.set_yticks([2, 2 ** (N - 1), 2 ** N])
    ax.set_yticklabels(["2", "$2^{" + str(N - 1) + "}$", "$2^{" + str(N) + "}$"])


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
            # ('new_inst_states', _plot_inst_states, '$E_N$ [GHz]'),
            # ('new_test_inst_states', _plot_inst_states_components, '$E_N$ [GHz]'),
            # ('new_entropy', _plot_entropy, r"$\mathcal{S}(\rho_{A})$"),
            ('new_magnetisation_2', _plot_magnetisation, r'$\mathcal{M}$'),
            # ('new_population', _plot_population, r'population'),
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
        f"12_BO_SHORT_STD_1D_std_",
        f"12_BO_SHORT_STD_2D_std_",
        f"12_BO_SHORT_STD_3D_std_",
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
