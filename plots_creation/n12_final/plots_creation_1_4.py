import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from numba import jit
from scipy import interpolate
from tqdm import tqdm
import quimb as q

import interaction_constants
from optimised_protocols import saver
from plots_creation.n12_final.utils import save_current_fig
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem

N = 12
LATTICE_SPACING = 1.5e-6

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

PLOTTED_LEGEND = False


@jit(nopython=True)
def get_figure_of_merit(state, state_index_1: int, state_index_2: int):
    density_matrix = state @ state.conjugate().transpose()
    rho_00 = density_matrix[state_index_1, state_index_1].real
    rho_11 = density_matrix[state_index_2, state_index_2].real
    off_diag_1 = density_matrix[state_index_2, state_index_1]
    off_diagonal = 2 * np.abs(off_diag_1).real
    result = (rho_00 + rho_11 + off_diagonal) / 2
    return result, rho_00, rho_11, off_diagonal


def _plot_protocol_and_fidelity(ax1: Axes, ax2: Axes, e_qs: EvolvingQubitSystem):
    Omega = interpolate.interp1d(e_qs.t_list, np.hstack((e_qs.Omega, e_qs.Omega[-1])), kind="previous",
                                 fill_value="extrapolate")
    Delta = interpolate.interp1d(e_qs.t_list, np.hstack((e_qs.Delta, e_qs.Delta[-1])), kind="previous",
                                 fill_value="extrapolate")

    Omega_color = "C0"
    Delta_color = "C3"

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1e6)))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1e9)))

    ax1.plot(e_qs.solved_t_list, [Omega(t) for t in e_qs.solved_t_list],
             color=Omega_color,
             label=r"$\Omega$",
             lw=3, ls='--', alpha=0.8)
    ax1.locator_params(nbins=3, axis='x')
    ax1.locator_params(nbins=4, axis='y')
    ax1.grid()

    Delta_ax = ax1
    Delta_ax.plot(e_qs.solved_t_list, [Delta(t) for t in e_qs.solved_t_list],
                  color=Delta_color,
                  label=r"$\Delta$",
                  lw=3, alpha=0.8)

    # ax1.set_ylabel(r"[GHz]")

    delta = (e_qs.t_list.max() - e_qs.t_list.min()) * 0.01
    ax1.set_xlim((e_qs.t_list.min() - delta, e_qs.t_list.max() + delta))

    # Panel 2
    ghz = e_qs.ghz_state.get_state_tensor()

    ghz_components = e_qs.ghz_state._get_components()
    state_index_1 = ghz_components[0].argmax()
    state_index_2 = ghz_components[1].argmax()

    # get_figure_of_merit(e_qs.solved_states[-1])
    # return

    foms = []
    rho_00s = []
    rho_11s = []
    off_diags = []
    # skip_i = 1
    # for _instantaneous_state in tqdm(e_qs.solved_states[::skip_i]):
    for _instantaneous_state in tqdm(e_qs.solved_states):
        fom, rho_00, rho_11, off_diag = get_figure_of_merit(_instantaneous_state, state_index_1, state_index_2)
        # print(q.fidelity(_instantaneous_state, ghz_components[0]), q.fidelity(_instantaneous_state, ghz_components[1]))

        foms.append(fom)
        rho_00s.append(rho_00)
        rho_11s.append(rho_11)
        off_diags.append(off_diag)

    ax2.plot(
        e_qs.solved_t_list,
        foms,
        lw=3,
        alpha=0.8,
        color="C2",
        zorder=2.1,
        label=r"$\mathcal{F}$",
    )

    ax2.plot(
        e_qs.solved_t_list,
        rho_00s,
        lw=2,
        alpha=0.8,
        color="C3",
        ls='--',
        label=r"$\rho_{00}$",
    )
    ax2.plot(
        e_qs.solved_t_list,
        rho_11s,
        lw=2,
        alpha=0.8,
        color="C4",
        ls='-.',
        label=r"$\rho_{11}$",
    )
    ax2.plot(
        e_qs.solved_t_list,
        off_diags,
        lw=2,
        alpha=0.8,
        color="C5",
        ls=':',
        label=r"$\rho_{01} + \rho_{10}$",
    )

    # ax2.set_ylabel(r"")
    ax2.set_xlabel(r"Time [$\upmu$s]")

    ax2.set_ylim((-0.1, 1.1))
    ax2.yaxis.set_ticks([0, 0.5, 1])
    ax2.grid()


def plot_BO_geometries_and_GHZ():
    plt.rcParams['axes.labelpad'] = 2

    gridspec_kwargs = {
        'nrows': 2,
        'ncols': 1,
        'hspace': 0.05,
        'top': 0.98, 'bottom': 0.18, 'left': 0.23, 'right': 0.97
    }
    gs = GridSpec(**gridspec_kwargs)

    for BO_file in BO_FILES:
        ghz = 'alt' if '_alt_' in BO_file else 'std'

        e_qs = saver.load(BO_file)

        fig = plt.figure(figsize=(3.5, 3))
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

        _plot_protocol_and_fidelity(
            ax1, ax2, e_qs,
        )

        ax1.text(
            0.02, 0.79,
            '[GHz]',
            horizontalalignment='left',
            verticalalignment='center',
            rotation='vertical',
            transform=fig.transFigure,
        )
        ax2.text(
            0.02, 0.37,
            r'$\mathcal{F}_\mathrm{' + ghz + '}$',
            horizontalalignment='left',
            verticalalignment='center',
            rotation='vertical',
            transform=fig.transFigure,
        )

        # ghz_ket = r"\ghzstdd" if "std" in BO_file else r"\ghzaltd"
        # ghz_ket += "{" + str(D) + "}"
        # ax1.set_title(f"${ghz_ket}$")

        ax1.get_xaxis().set_visible(False)
        save_current_fig(f"_new_protocol_and_fidelity_{BO_file}")

        # ax2.legend(loc=9, framealpha=1)
        # save_current_fig(f"_legend_new_protocol_and_fidelity_{BO_file}")
        # raise RuntimeError()


if __name__ == '__main__':
    BO_FILES = [
        # f"12_BO_SHORT_STD_1D_std_",
        # f"12_BO_SHORT_STD_2D_std_",
        # f"12_BO_SHORT_STD_3D_std_",
        # f"12_BO_COMPARE_BO_WIDER_1D_alt_",
        # f"12_BO_COMPARE_BO_2D_alt_",
        # f"12_BO_COMPARE_BO_3D_alt_",
        # "entanglement_entropy_ramp__8_2D_std",
        "entanglement_entropy_ramp_2_8_2D_std",
        "entanglement_entropy_ramp__12_2D_std",
        # "entanglement_entropy_ramp__16_2D_std",
        "entanglement_entropy_ramp__8_2D_alt",
        "entanglement_entropy_ramp__12_2D_alt",
        # "entanglement_entropy_ramp__16_2D_alt",
    ]
    plot_BO_geometries_and_GHZ()
