import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import quimb as q
from matplotlib import ticker
from matplotlib.animation import FFMpegFileWriter, FFMpegWriter
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

import interaction_constants
from faster_ffmpeg_writer import FasterFFMpegWriter
from protocol_generator.interpolation_pg import InterpolationPG
from qubit_system.geometry import RegularLattice
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem
from qubit_system.utils.ghz_states import CustomGHZState
from statistics_plot import get_e_qs

COLORMAP = 'plasma'
norm = LogNorm(vmin=1e-4, vmax=1, clip=True)


def setup_figure():
    fig = plt.figure(figsize=(12, 9))
    gridspec_kwargs = {
        'nrows': 4,
        'ncols': 2,
        'height_ratios': [4, 1, 8, 1],
        'width_ratios': [20, 1],
        'top': 0.95,
        'bottom': 0.06,
        'left': 0.1,
        'right': 0.92,
        'wspace': 0.05,
        'hspace': 0.05

    }
    gs = GridSpec(**gridspec_kwargs)
    cax = fig.add_subplot(gs[2:, 1])

    norm = LogNorm(vmin=1e-4, vmax=1, clip=True)
    mappable = ScalarMappable(norm=norm, cmap=COLORMAP)
    cbar = plt.colorbar(mappable, cax=cax, ticks=[1e-4, 1e-3, 1e-2, 1e-1, 1], extend='min')
    cbar.ax.set_yticklabels(['$< 0.0001$', '$0.001$', '$0.01$', '$0.1$', '$1$'])
    cbar.ax.set_ylabel(r"Eigenstate population ${ \left|\langle \psi (t) | \psi_k (t) \rangle \right| }^2$")

    ax1 = fig.add_subplot(gs[2, 0])

    ax1.set_ylim((1e-12, 1))
    ax1.set_yscale('log', basey=10)

    for label in ax1.get_xticklabels(which="both"):
        label.set_visible(False)
    ax1.get_xaxis().get_offset_text().set_visible(False)
    ax1.set_xlabel("")

    ax2 = fig.add_subplot(gs[3, 0], sharex=ax1)

    ax2.set_ylim((0, 1e-12))
    ax2.set_yticks([0])
    ax2.xaxis.set_major_formatter(ticker.EngFormatter('Hz'))

    fig.suptitle(f"Instantaneous eigenstates $\psi_k (t)$ over time")
    xlabel = 'Eigenstate energy $E$'
    ylabel = r'Fidelity with GHZ state ${\left| \langle \mathrm{GHZ} | \psi_k (t) \rangle \right| }^2$'
    label_kwargs = {'weight': 'bold'}
    ax1.set_ylabel(ylabel)
    ax2.set_xlabel(xlabel)
    # fig.text(0.5, 0.03, xlabel, ha='center', va='center', **label_kwargs)
    # fig.text(0.03, 0.5, ylabel, ha='center', va='center', rotation='vertical', **label_kwargs)

    ax3 = fig.add_subplot(gs[0, :])
    e_qs.plot_Omega_and_Delta(ax3, plot_title=False)
    ax3.grid()

    return fig, ax1, ax2, ax3


def update_figure(i: int, e_qs: EvolvingQubitSystem, ax1: Axes, ax2: Axes, ax3: Axes):
    t = e_qs.solved_t_list[i]
    Omega = e_qs.Omega[i]
    Delta = e_qs.Delta[i]
    hamiltonian = e_qs.get_hamiltonian(Omega, Delta)
    dense_hamiltonian = q.quimbify(hamiltonian, sparse=False)
    eigenenergies, inst_eigenstates = q.eigh(dense_hamiltonian)

    inst_eigenstates = inst_eigenstates.transpose()
    # Eigenstate returned in columns. Transpose so they're in rows for "for loop" below.

    system_state = q.quimbify(e_qs.solved_states[i], sparse=False)
    ghz_state = q.quimbify(e_qs.ghz_state.get_state_tensor(), sparse=False)

    energies = []
    ghz_overlaps = []
    instantaneous_eigenstate_populations = []

    counts = defaultdict(lambda: 1)
    for i, instantaneous_eigenstate in enumerate(inst_eigenstates):
        eigenenergy = eigenenergies[i]
        # if eigenenergy > 0.5e9:
        #     continue

        ghz_overlap = q.fidelity(ghz_state, instantaneous_eigenstate)
        instantaneous_state_overlap = q.fidelity(system_state, instantaneous_eigenstate)

        try:
            _found = False
            for energy_i in range(len(energies)):
                isclose = partial(np.isclose, atol=1e-15, rtol=1e-10)
                if isclose(energies[energy_i], eigenenergy) \
                        and isclose(ghz_overlaps[energy_i], ghz_overlap) \
                        and isclose(instantaneous_eigenstate_populations[energy_i], instantaneous_state_overlap):
                    counts[energy_i] += 1
                    # print(energies[energy_i], eigenenergy)
                    # print(ghz_overlaps[energy_i], ghz_overlap)
                    # print(instantaneous_eigenstate_populations[energy_i], instantaneous_state_overlap)
                    # print("213")
                    _found = True
                    break
            if _found:
                continue
        except ValueError:
            pass
        energies.append(eigenenergy)
        ghz_overlaps.append(ghz_overlap)
        instantaneous_eigenstate_populations.append(instantaneous_state_overlap)

    count_labels = []
    for i in range(len(energies)):
        _count = counts[i]
        energy = energies[i]
        ghz_overlap = ghz_overlaps[i]
        text = ax1.text(energy, ghz_overlap, str(_count), ha='center', va='center', zorder=10000)
        count_labels.append(text)


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

    cmap_array = ListedColormap(cmap_array)
    norm = LogNorm(vmin=1e-4, vmax=1, clip=True)

    scatter1 = ax1.scatter(
        energies, ghz_overlaps,
        c=instantaneous_eigenstate_populations,
        cmap=cmap_array, norm=norm,
        s=10
    )

    scatter2 = ax2.scatter(
        energies_below_lim, ghz_overlaps_below_lim,
        c=instantaneous_eigenstate_populations_below_lim,
        cmap=cmap_array, norm=norm,
        s=10
    )
    ax1.set_ylabel(f"$t = {t * 1e6:.2f}$ $\mu$s")
    # text = ax1.text(0.95, 0.05, f"$t = {t * 1e6:.2f}$ $\mu$s", ha='right', va='center', transform=ax1.transAxes)

    line = ax3.axvline(x=t, alpha=0.3, color='k', linewidth=3)

    return [scatter1, scatter2, line] + count_labels


def save_video(e_qs: EvolvingQubitSystem, name: str):
    fig, ax1, ax2, ax3 = setup_figure()

    n = len(e_qs.Omega)
    indices = np.arange(n, step=1)

    # Setup x limits
    for i in tqdm(indices):
        plots = update_figure(i, e_qs, ax1, ax2, ax3)
        for plot in plots:
            plot.remove()
    print("Setup limits")

    # moviewriter = FFMpegWriter(fps=60)
    moviewriter = FasterFFMpegWriter(fps=30, bitrate=3000)
    with moviewriter.saving(fig, f'plots/final/videos/___eigenstates_evolution_{name}.mp4', dpi=90):
        for i in tqdm(indices):
            plots = update_figure(i, e_qs, ax1, ax2, ax3)
            moviewriter.grab_frame()
            for plot in plots:
                plot.remove()

    print(f"Saved video for: {name}")


if __name__ == '__main__':
    plt.rc('animation', ffmpeg_path=str(Path(__file__).parent / "ffmpeg" / "ffmpeg"))
    plt.rc('text', usetex=True)
    plt.rc('font', family="serif", serif="CMU Serif")
    plt.rc('text.latex', preamble=r'\usepackage{upgreek}')

    setups = [
        # (1, "std"),
        # (1, "alt"),
        # (2, "std"),
        # (2, "alt"),
        (3, "std"),
        (3, "alt"),
    ]
    for D, ghz in setups:
        e_qs = get_e_qs(D, ghz)
        name = f"{D}_{ghz}_{e_qs.get_fidelity_with('ghz'):.5f}"

        save_video(e_qs, name)
