import matplotlib.pyplot as plt
import numpy as np
import quimb as q
from matplotlib import ticker
from matplotlib.axes import Axes
from scipy import interpolate
from tqdm import tqdm

import interaction_constants
from optimised_protocols import saver
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem
from plots_creation.n12_final.utils import save_current_fig
from qubit_system.utils.states import get_product_basis_states_index

N = 12
LATTICE_SPACING = 1.5e-6

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

PLOTTED_LEGEND = False


def _plot_protocol_and_fidelity(ax1: Axes, ax2: Axes, e_qs: EvolvingQubitSystem, with_antisymmetric_ghz: bool,
                                first: bool, last: bool, first_row: bool):
    Omega = interpolate.interp1d(e_qs.t_list, np.hstack((e_qs.Omega, e_qs.Omega[-1])), kind="previous",
                                 fill_value="extrapolate")
    Delta = interpolate.interp1d(e_qs.t_list, np.hstack((e_qs.Delta, e_qs.Delta[-1])), kind="previous",
                                 fill_value="extrapolate")

    Omega_color = "C0"
    Delta_color = "C3"

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1e6)))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1e9)))

    ax1.plot(e_qs.solved_t_list, [Omega(t) for t in e_qs.solved_t_list], color=Omega_color, lw=3, alpha=0.8)
    ax1.locator_params(nbins=3, axis='x')

    Delta_ax = ax1
    Delta_ax.plot(e_qs.solved_t_list, [Delta(t) for t in e_qs.solved_t_list], color=Delta_color, lw=3, alpha=0.8)

    ax1.locator_params(nbins=4, axis='y')
    if first:
        ax1.set_ylabel(r"[GHz]")

    delta = (e_qs.t_list.max() - e_qs.t_list.min()) * 0.01
    ax1.set_xlim((e_qs.t_list.min() - delta, e_qs.t_list.max() + delta))

    # Panel 2
    ghz = e_qs.ghz_state.get_state_tensor()

    ghz_components = e_qs.ghz_state._get_components()
    state_index_1 = ghz_components[0].argmax()
    state_index_2 = ghz_components[1].argmax()

    def get_figure_of_merit(state):
        density_matrix = state @ state.H
        # print(density_matrix.shape)
        rho_00 = density_matrix[state_index_1, state_index_1]
        rho_11 = density_matrix[state_index_2, state_index_2]
        off_diag_1 = density_matrix[state_index_2, state_index_1]
        off_diag_2 = density_matrix[state_index_1, state_index_2]
        off_diagonal = np.abs(off_diag_1 + off_diag_2)
        result = (rho_00 + rho_11 + off_diagonal) / 2
        diag = (rho_11 * rho_00) ** 0.5
        abs_offdiag = abs(off_diag_1)
        print(diag, abs_offdiag, diag==abs_offdiag)
        print(result, rho_00, rho_11, off_diag_1, off_diag_2)

        # print(state.shape)
        # state_2 = state[state_index_2, 0]
        # state_1 = state[state_index_1, 0]
        # print(state_1 ** 2, state_2 ** 2, state_2 * state_1, state_1, state_2, 'what')

        ghz_fidelity = q.fidelity(state, ghz)
        print(result, ghz_fidelity)
        return result

    get_figure_of_merit(e_qs.solved_states[-1])
    return

    foms = []
    skip_i = 1
    for _instantaneous_state in tqdm(e_qs.solved_states[::skip_i]):
        foms.append(get_figure_of_merit(_instantaneous_state))

    ax2.plot(
        e_qs.solved_t_list[::skip_i],
        foms,
        lw=1,
        alpha=0.8,
        color="C2",
    )
    if not first_row:
        ax2.set_xlabel(r"[$\upmu$s]")

    ax2.set_ylim((-0.1, 1.1))
    ax2.yaxis.set_ticks([0, 0.5, 1])
    ax2.grid()


def plot_BO_geometries_and_GHZ():
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(5, 3, wspace=0.3, hspace=0.05, height_ratios=[1, 1, 0.5, 1, 1],
                          top=0.93, bottom=0.09, left=0.08, right=0.98)
    # fig, (axs) = plt.subplots(4, 3, sharex='col', figsize=(16, 6))
    e_qs = None
    for col, D in enumerate([1, 2, 3]):
        for row, ghz in enumerate(["std", "alt"]):
            BO_file = f"12_BO_COMPARE_BO_{D}D_{ghz}_"

            # if e_qs is None:
            #     e_qs = saver.load(BO_file)
            e_qs = saver.load(BO_file)

            row_ = row * 3
            ax1 = fig.add_subplot(gs[row_, col])
            ax2 = fig.add_subplot(gs[row_ + 1, col], sharex=ax1)
            _plot_protocol_and_fidelity(
                ax1, ax2, e_qs, with_antisymmetric_ghz=True,
                first=col == 0, last=col == 2,
                first_row=row_ == 0
            )
            ghz_ket = r"\ghzstd" if ghz == "std" else r"\ghzalt"
            ax1.set_title(f"{D}D,  " + r"$\quad \ket{\psi_{\mathrm{target}}} = " + ghz_ket + "$")
            ax1.get_xaxis().set_visible(False)
    # save_current_fig("new_fid_bo_geometries_ghz_protocol_and_fidelity")


if __name__ == '__main__':
    plot_BO_geometries_and_GHZ()
