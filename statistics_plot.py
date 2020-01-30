import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap, LogNorm
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import quimb as q

import interaction_constants
from protocol_generator.interpolation_pg import InterpolationPG
from qubit_system.geometry import RegularLattice
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem
from qubit_system.utils.ghz_states import CustomGHZState
from qubit_system.utils import states

COLORMAP = 'viridis'


def get_e_qs(D: int, GHZ: str):
    N_RYD = 50
    C6 = interaction_constants.get_C6(N_RYD)

    LATTICE_SPACING = 1.5e-6

    N = 8
    assert GHZ == "std" or GHZ == "alt", f"GHZ has to be std or alt, not {GHZ}"
    ghz_component = [True, True, True, True, True, True, True, True] if GHZ == "std" else None
    if D == 1:
        geometry_shape = (8,)
        if GHZ == "alt":
            ghz_component = [True, False, True, False, True, False, True, False]
    elif D == 2:
        geometry_shape = (4, 2)
        if GHZ == "alt":
            ghz_component = [True, False, False, True, True, False, False, True]
    elif D == 3:
        geometry_shape = (2, 2, 2)
        if GHZ == "alt":
            ghz_component = [True, False, False, True, False, True, True, False]
    else:
        raise ValueError(f"Unknown D: {D}")

    geometry = RegularLattice(shape=geometry_shape, spacing=LATTICE_SPACING)
    ghz_state = CustomGHZState(N, ghz_component)
    protocol = optimised_protocols[D][GHZ]

    t = 2e-6
    interpolation_timesteps = 3000
    t_list = np.linspace(0, t, interpolation_timesteps + 1)

    protocol_generator = InterpolationPG(t_list, kind="quadratic")
    Omega, Delta = protocol_generator.get_protocol(np.array(protocol))
    e_qs = EvolvingQubitSystem(
        N, C6, geometry,
        Omega, Delta,
        t_list,
        ghz_state=ghz_state
    )
    start_time = time.time()
    e_qs.solve()

    print(f"Solved in {time.time() - start_time:.3f}s")
    ghz_fidelity = e_qs.get_fidelity_with("ghz")

    print(f"{D}D {GHZ} fidelity: {ghz_fidelity:.5f}")
    return e_qs


def plot_eigenstate_populations(e_qs: EvolvingQubitSystem, ax: Axes, log: bool):
    states_list = states.get_states(e_qs.N)

    fidelities = []
    unique_labels = set()
    for i, state in enumerate(tqdm(states_list)):
        label = states.get_label_from_state(state)
        state_product_basis_index = states.get_product_basis_states_index(state)
        state_fidelities = [np.abs(_instantaneous_state.flatten()[state_product_basis_index]) ** 2
                            for _instantaneous_state in e_qs.solved_states]

        is_excited = [states.is_excited(_qubit) for _qubit in state]

        if all(is_excited[i] == e_qs.ghz_state.single_component[i] for i in range(e_qs.N)) \
                or all(is_excited[i] != e_qs.ghz_state.single_component[i] for i in range(e_qs.N)):
            fidelities.append(state_fidelities)

            plot_label = r"$P_{\textrm{GHZ component}}$"
            color = 'r'
            linewidth = 1
        elif 'e' not in label:
            plot_label = r"$P_{" + f"{label.upper()[0]}" + "}$"
            color = 'g'
            linewidth = 1
        else:
            plot_label = 'Others'
            color = 'k'
            linewidth = 0.5

        if plot_label in unique_labels:
            plot_label = None
        else:
            unique_labels.add(plot_label)

        ax.plot(
            e_qs.solved_t_list,
            state_fidelities,
            label=plot_label,
            color=color,
            linewidth=linewidth,
            alpha=0.5
        )

    fidelities_sum = np.array(fidelities).sum(axis=0)
    ax.plot(e_qs.solved_t_list, fidelities_sum,
            label=r"$\sum P_{\textrm{GHZ component}}$",
            color='C0', linestyle=":", linewidth=1, alpha=0.7)

    ax.set_ylabel("Population")
    # ax.set_title("Basis state populations")
    ax.yaxis.set_ticks([0, 0.5, 1])
    if log:
        ax.set_yscale('log', basey=10)
        ax.set_ylim((1e-4, 2))
    else:
        ax.set_ylim((-0.1, 1.1))

    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda x: x[0]))
    ax.legend(handles, labels)


def plot_n_pc(e_qs: EvolvingQubitSystem, ax: Axes):
    system_states = e_qs.solved_states
    N_pcs = []
    for state in tqdm(system_states):
        sum_powers = np.sum((np.power(np.abs(state), 4)))
        N_pc = 1 / sum_powers
        N_pcs.append(N_pc)
    ax.plot(
        e_qs.solved_t_list, N_pcs,
        color='C0', linewidth=1
    )

    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.axhline(1, color='k', linewidth=0.5, alpha=0.5)
    ax.axhline(2, color='k', linewidth=0.5, alpha=0.5)

    # ax.set_title("Number of Principal Components")
    ax.set_ylabel("$N_{PC}$")


def generate_evolution_plot(e_qs: EvolvingQubitSystem, name: str):
    fig, axs = plt.subplots(5, 1, sharex='all', figsize=(12, 9))

    e_qs.plot_Omega_and_Delta(axs[0], plot_title=False)
    e_qs.plot_ghz_states_overlaps(axs[1], plot_title=False, with_antisymmetric_ghz=False)
    plot_eigenstate_populations(e_qs, axs[2], log=True)
    plot_n_pc(e_qs, axs[3])
    e_qs.plot_entanglement_entropies(axs[4], plot_title=False)

    plt.xlabel('Time')
    plt.tight_layout()

    for ax in axs:
        ax.grid()

        # Move legend to the right
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    save_fig(fig, name + "_evolution")


def generate_eigenstates_plot(e_qs: EvolvingQubitSystem, name: str):
    ts = [0.05e-6, 0.5e-6, 1e-6, 1.5e-6, 1.95e-6]
    solved_t_list = np.array(e_qs.solved_t_list)
    indices = [
        (np.abs(solved_t_list - _t)).argmin()
        for _t in ts
    ]
    # indices = [0, -1]
    indices.insert(0, 0)
    indices.append(-1)

    ghz_state = q.quimbify(e_qs.ghz_state.get_state_tensor(), sparse=False)

    # print(indices)
    fig = plt.figure(figsize=(12, 9))
    gridspec_kwargs = {
        'nrows': len(indices) * 3,
        'ncols': 2,
        'height_ratios': [8, 1, 1] * len(indices),
        'width_ratios': [20, 1],
        'top': 0.95,
        'bottom': 0.06,
        'left': 0.1,
        'right': 0.92,
        'wspace': 0.05,
        'hspace': 0.05

    }
    gs = GridSpec(**gridspec_kwargs)
    first_axes = None
    for _i, i in enumerate(indices):
        t = solved_t_list[i]
        Omega = e_qs.Omega[i]
        Delta = e_qs.Delta[i]
        hamiltonian = e_qs.get_hamiltonian(Omega, Delta)
        dense_hamiltonian = q.quimbify(hamiltonian, sparse=False)
        eigenenergies, inst_eigenstates = q.eigh(dense_hamiltonian)
        inst_eigenstates = inst_eigenstates.transpose()  # Eigenstate returned in columns. Transpose so they're in rows for "for loop" below.

        system_state = q.quimbify(e_qs.solved_states[i], sparse=False)
        energies = []
        ghz_overlaps = []
        instantaneous_eigenstate_populations = []
        for i, instantaneous_eigenstate in enumerate(inst_eigenstates):
            eigenenergy = eigenenergies[i]
            energies.append(eigenenergy)
            ghz_overlap = q.fidelity(ghz_state, instantaneous_eigenstate)
            ghz_overlaps.append(ghz_overlap)

            instantaneous_state_overlap = q.fidelity(system_state, instantaneous_eigenstate)
            # print(instantaneous_state_overlap)
            instantaneous_eigenstate_populations.append(instantaneous_state_overlap)

        energies = np.array(energies)
        ghz_overlaps = np.array(ghz_overlaps)
        # print(1, ghz_overlaps.sum())
        instantaneous_eigenstate_populations = np.array(instantaneous_eigenstate_populations)
        # print(2, instantaneous_eigenstate_populations.sum())

        # Sort by instantaneous eigenstate population
        sort = np.argsort(instantaneous_eigenstate_populations)
        energies = energies[sort]
        ghz_overlaps = ghz_overlaps[sort]
        instantaneous_eigenstate_populations = instantaneous_eigenstate_populations[sort]

        # Clip minimum population to 1e-12 to avoid masking
        instantaneous_eigenstate_populations[instantaneous_eigenstate_populations < 1e-16] = 1e-16

        above_limit = ghz_overlaps > 1e-12
        energies, energies_below_lim = energies[above_limit], energies[~above_limit]
        ghz_overlaps, ghz_overlaps_below_lim = ghz_overlaps[above_limit], ghz_overlaps[~above_limit]
        instantaneous_eigenstate_populations, instantaneous_eigenstate_populations_below_lim = \
            instantaneous_eigenstate_populations[above_limit], instantaneous_eigenstate_populations[~above_limit]

        # Get the colormap colors
        cmap = plt.cm.get_cmap(COLORMAP)
        cmap_array = cmap(np.arange(cmap.N))

        cmap_array[:, -1] = np.linspace(0.3, 1, cmap.N)

        # Create new colormap
        cmap_array = ListedColormap(cmap_array)
        # norm = Normalize(vmin=-3, vmax=0, clip=True)
        norm = LogNorm(vmin=1e-4, vmax=1, clip=True)

        # ax1 = axs[_i * 2, 0]
        axes_row = _i * 3
        if first_axes:
            ax1 = fig.add_subplot(gs[axes_row, 0], sharex=first_axes)
        else:
            ax1 = fig.add_subplot(gs[axes_row, 0])
            first_axes = ax1
        ax2 = fig.add_subplot(gs[axes_row + 1, 0], sharex=ax1)

        ax1.scatter(
            energies, ghz_overlaps,
            c=instantaneous_eigenstate_populations,
            cmap=cmap_array, norm=norm,
            s=10
        )
        # ax.set_title(f"$t = {t * 1e6:.2f} \,$ $\mu$s")
        ax1.set_ylabel(f"$t = {t * 1e6:.2f}$ $\mu$s")
        ax1.set_ylim((1e-12, 1))
        ax1.set_yscale('log', basey=10)
        # ax1.set_ylim((0, 1))

        ax2.scatter(
            energies_below_lim, ghz_overlaps_below_lim,
            c=instantaneous_eigenstate_populations_below_lim,
            cmap=cmap_array, norm=norm,
            s=10
        )
        ax2.set_ylim((0, 1e-12))
        ax2.set_yticks([0])

        if _i != len(indices) - 1:
            axes_to_clear = [ax1, ax2]
        else:
            axes_to_clear = [ax1]
        for ax in axes_to_clear:
            for label in ax.get_xticklabels(which="both"):
                label.set_visible(False)
            ax.get_xaxis().get_offset_text().set_visible(False)
            ax.set_xlabel("")

    for label in ax2.get_xticklabels(which="both"):
        label.set_visible(True)
    ax2.get_xaxis().get_offset_text().set_visible(True)
    ax2.xaxis.set_major_formatter(ticker.EngFormatter('Hz'))

    fig.suptitle(f"Instantaneous eigenstates $\psi_k (t)$s at increasing $t$")
    xlabel = 'Eigenstate energy $E$'
    ylabel = r'Fidelity with GHZ state ${\left| \langle \mathrm{GHZ} | \psi_k (t) \rangle \right| }^2$'
    label_kwargs = {'weight': 'bold'}
    fig.text(0.5, 0.03, xlabel, ha='center', va='center', **label_kwargs)
    fig.text(0.03, 0.5, ylabel, ha='center', va='center', rotation='vertical', **label_kwargs)

    cax = fig.add_subplot(gs[:, 1])
    # mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable = ScalarMappable(norm=norm, cmap=COLORMAP)
    # cbar = plt.colorbar(mappable, cax=cax, ticks=[-3, -2, -1, 0], extend='min')
    # cbar.ax.set_yticklabels(['$< -3$', '$-2$', '$-1$', '$0$'])
    # cbar.ax.set_ylabel(r"Eigenstate population $\log_{10} {\left|\langle \psi (t) | \psi_k (t) \rangle \right| }^2$")
    # cbar = plt.colorbar(mappable, cax=cax, ticks=[1e-3, 1e-2, 1e-1, 1], extend='min')
    # cbar.ax.set_yticklabels(['$< 0.001$', '$0.01$', '$0.1$', '$1$'])
    cbar = plt.colorbar(mappable, cax=cax, ticks=[1e-4, 1e-3, 1e-2, 1e-1, 1], extend='min')
    cbar.ax.set_yticklabels(['$< 0.0001$', '$0.001$', '$0.01$', '$0.1$', '$1$'])
    cbar.ax.set_ylabel(r"Eigenstate population ${ \left|\langle \psi (t) | \psi_k (t) \rangle \right| }^2$")
    save_fig(fig, name + "_eigenstate")


def generate_N_pc_investigation_plot(e_qs: EvolvingQubitSystem, name: str):
    fig = plt.figure(figsize=(15, 9))
    gridspec_kwargs = {
        'nrows': 2,
        'ncols': 2,
        'height_ratios': [2, 5],
        'width_ratios': [40, 1],
        'top': 0.95,
        'bottom': 0.05,
        'left': 0.05,
        'right': 0.95,
        'wspace': 0.05,
        'hspace': 0.1

    }
    gs = GridSpec(**gridspec_kwargs)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    system_states = e_qs.solved_states
    N_pcs = []

    limits = [1e-4, 1e-3, 1e-2, 1e-1]
    counts = defaultdict(list)

    cs = []
    for i, state in enumerate(tqdm(system_states)):
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

    ax1.plot(
        e_qs.solved_t_list, N_pcs,
        color='C0', linewidth=1
    )

    ax1.set_ylim([0, ax1.get_ylim()[1]])
    ax1.axhline(1, color='k', linewidth=0.5, alpha=0.5)
    ax1.axhline(2, color='k', linewidth=0.5, alpha=0.5)

    ax1.set_ylabel("$N_{PC}$")

    norm = LogNorm(vmin=1e-3, vmax=1, clip=True)
    cs = np.array(cs).transpose()
    ax2.imshow(
        cs,
        aspect='auto',
        cmap=plt.cm.get_cmap(COLORMAP), norm=norm,
        origin='lower',
        extent=(0, e_qs.solved_t_list[-1], 0, 256)
    )
    ax2.set_ylabel(r"Eigenstate population ${ \left|\langle \psi (t) | \psi_k (t) \rangle \right| }^2$")

    mappable = ScalarMappable(norm=norm, cmap=COLORMAP)
    cax = fig.add_subplot(gs[1, 1])
    cbar = plt.colorbar(mappable, cax=cax, ticks=[1e-3, 1e-2, 1e-1, 1], extend='min')
    cbar.ax.set_yticklabels(['$< 0.001$', '$0.01$', '$0.1$', '$1$'])

    ax2.set_ylim((0, 256))

    # for limit in limits:
    #     ax2.plot(
    #         e_qs.solved_t_list, counts[limit],
    #         label=limit
    #     )
    # ax2.legend()

    ax1.xaxis.set_major_formatter(ticker.EngFormatter('s'))
    ax2.set_xlabel('Time')
    save_fig(fig, name + "_n_pc_2")


def save_fig(fig, name: str):
    plt.show()
    # fig.savefig(f"plots/final/2_{name}.png", dpi=300)
    plt.close(fig)


optimised_protocols = {
    1: {
        'std': [2.65629869e+07, 1.10137775e+09, 2.10554803e+07, 1.53833774e+09, 1.43818374e+09, 1.15576644e+09],
        # 0.96210
        # 'std': [7.22461680e+08, 6.49773034e+08, 6.00660122e+08, 1.13152837e+09, 1.52233991e+09, 1.30382513e+09],
        # 0.90795
        # 'std': [9.28880828e+08, 1.03690960e+09, 8.49647496e+08, 1.81531908e+09, 9.70763685e+08, 1.22157560e+09],
        # 0.89575
        # 'alt': [1034883.35720177, 10746002.32511696, 13138604.21549956, 12611089.34283306, 14807475.81352524,
        #         12823830.46326383],
        # 0.79958
        'alt': [2565801.10085787, 12372232.59839516, 6488618.09446081, 19508115.86734799, 13904167.59376139,
                10511527.37629483]
        # 0.87312
    },
    2: {
        # 'std': [1.59175109e+09, 4.40798493e+08, 8.22430687e+08, 1.52515077e+09, 2.72788764e+09, 2.08805395e+09],
        # 0.60493
        'std': [1.65811738e+09, 8.29998385e+08, 1.82769297e+09, 2.57907175e+09, 2.71117607e+09, 1.94975775e+09],
        # 0.97098
        # 'alt': [8.94101353e+07, 1.34436283e+08, 3.17347152e+07, 1.90844269e+08, 9.70544131e+07, 8.64859020e+07]
        # 0.99998
        'alt': [1.64782808e+07, 9.85419648e+07, 8.34900676e+07, 8.67334467e+07, 2.05864069e+08, 1.06214334e+08]
    },
    3: {
        'std': [1.94735741e+09, 1.17085412e+09, 1.98626869e+09, 3.46749476e+09, 1.84125209e+09, 2.39486724e+09],
        'alt': [1.67533312e+08, 1.80310100e+08, 8.37019945e+06, 3.61327852e+08, 1.92993336e+08, 1.63023464e+08]
    }
}

if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family="serif", serif="CMU Serif")
    plt.rc('text.latex', preamble=r'\usepackage{upgreek}')

    setups = [
        # (1, "std"),
        # (1, "alt"),
        # (2, "std"),
        (2, "alt"),
        # (3, "std"),
        # (3, "alt"),
    ]
    for D, ghz in setups:
        e_qs = get_e_qs(D, ghz)
        name = f"{D}_{ghz}_{e_qs.get_fidelity_with('ghz'):.5f}"
        generate_evolution_plot(e_qs, name)
        generate_eigenstates_plot(e_qs, name)
        # generate_N_pc_investigation_plot(e_qs, name)
