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
from protocol_generator.custom_interpolation_pg import CustomInterpolationPG
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem

N = 12
LATTICE_SPACING = 1.5e-6

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

PLOTTED_LEGEND = False

protocols = {
    # (1, "std", 0.5): [1.08081279e+09, 3.98468827e+08, 4.95220757e+08, 9.39029281e+08, 1.60076484e+09, 1.39668882e+09],
    # (1, "alt", 0.8): [12864164.00251228, 13929999.5630103, 16182438.28004114, 18748875.50105673, 19836048.07283583,
    #                   11165417.52171274],

    # (1, "std", 0.2): [1.21344545e+09, 1.23152656e+09, 1.10389513e+08, 1.02223011e+09, 1.86227838e+09, 1.20940030e+09],
    # (1, "alt", 0.2): [441436.76288613, 17958550.53149551, 11187935.55716398, 24163677.64513641, 15150961.32302503,
    #              10424052.76692684],
    # (2, "std", 0.2): [2.00130092e+09, 1.57667386e+09, 1.36307468e+09, 2.27962022e+09, 2.16387636e+09, 2.19094970e+09],
    "12_BO_COMPARE_BO_2D_alt_": ([3.51832094e+06, 1.65642207e+08, 8.60534422e+07, 2.14008038e+08, 1.61479519e+08, 1.02126899e+08], 0.2),
    # (3, "std", 0.2): [1.82728899e+09, 1.70821665e+09, 2.13164349e+09, 2.33794257e+09, 3.77499293e+09, 2.89800233e+09],
    # (3, "alt", 0.2): [2.70546737e+08, 1.33471648e+08, 7.25294404e+07, 3.29288148e+08, 2.67062494e+08, 2.03814321e+08],
    "12_BO_SHORT_STD_1D_std_": (
    [4.48029301e+08, 7.19530539e+08, 1.72819439e+08, 9.14124784e+08, 1.33494897e+09, 1.20572155e+09], 0.8),
    "12_BO_SHORT_STD_2D_std_": (
    [2.03656556e+09, 1.70305148e+09, 1.58486578e+09, 1.44662268e+09, 2.67199684e+09, 2.26869872e+09], 0.8),
    "12_BO_SHORT_STD_3D_std_": (
    [1.99357238e+09, 1.89712671e+09, 1.41245085e+08, 3.42986973e+09, 2.68313113e+09, 2.96504632e+09], 0.8),
}


@jit(nopython=True)
def get_figure_of_merit(state, state_index_1: int, state_index_2: int):
    density_matrix = state @ state.conjugate().transpose()
    rho_00 = density_matrix[state_index_1, state_index_1].real
    rho_11 = density_matrix[state_index_2, state_index_2].real
    off_diag_1 = density_matrix[state_index_2, state_index_1]
    off_diagonal = 2 * np.abs(off_diag_1).real
    result = (rho_00 + rho_11 + off_diagonal) / 2
    return result, rho_00, rho_11, off_diagonal


def _plot_protocol_and_fidelity(ax1: Axes, ax2: Axes, e_qs: EvolvingQubitSystem, BO_file):
    omegas_ = []
    deltas_ = []
    foms_ = []
    rho_00s_ = []
    rho_11s_ = []
    off_diags_ = []
    input_, tukey_alpha = protocols[BO_file]
    pg = CustomInterpolationPG(e_qs.t_list, kind="cubic", tukey_alpha=tukey_alpha, noise=0.05)
    for i in range(20):
        Omega, Delta = pg.get_protocol(np.array(input_))
        e_qs.Omega = Omega
        e_qs.Delta = Delta
        e_qs.solve()

        Omega = interpolate.interp1d(e_qs.t_list, np.hstack((e_qs.Omega, e_qs.Omega[-1])), kind="previous",
                                     fill_value="extrapolate")
        Delta = interpolate.interp1d(e_qs.t_list, np.hstack((e_qs.Delta, e_qs.Delta[-1])), kind="previous",
                                     fill_value="extrapolate")

        omegas_.append([Omega(t) for t in e_qs.solved_t_list])
        deltas_.append([Delta(t) for t in e_qs.solved_t_list])

        ghz_components = e_qs.ghz_state._get_components()
        state_index_1 = ghz_components[0].argmax()
        state_index_2 = ghz_components[1].argmax()

        foms = []
        rho_00s = []
        rho_11s = []
        off_diags = []
        foms_skip = 30
        for _instantaneous_state in tqdm(e_qs.solved_states[::foms_skip]):
            fom, rho_00, rho_11, off_diag = get_figure_of_merit(_instantaneous_state, state_index_1, state_index_2)
            foms.append(fom)
            rho_00s.append(rho_00)
            rho_11s.append(rho_11)
            off_diags.append(off_diag)
        foms_.append(foms)
        rho_00s_.append(rho_00s)
        rho_11s_.append(rho_11s)
        off_diags_.append(off_diags)

    omegas_ = np.array(omegas_)
    deltas_ = np.array(deltas_)
    foms_ = np.array(foms_)
    rho_00s_ = np.array(rho_00s_)
    rho_11s_ = np.array(rho_11s_)
    off_diags_ = np.array(off_diags_)

    Omega_color = "C0"
    Delta_color = "C3"

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1e6)))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1e9)))

    ax1.locator_params(nbins=3, axis='x')
    ax1.locator_params(nbins=4, axis='y')
    ax1.grid()

    ax1.plot(
        e_qs.solved_t_list,
        np.mean(omegas_, axis=0),
        color=Omega_color,
        label=r"$\Omega$",
        lw=3, ls='--', alpha=0.8
    )
    ax1.fill_between(
        e_qs.solved_t_list,
        np.quantile(omegas_, 0.25, axis=0),
        np.quantile(omegas_, 0.75, axis=0),
        color=Omega_color, alpha=0.3
    )

    Delta_ax = ax1
    Delta_ax.plot(
        e_qs.solved_t_list,
        np.mean(deltas_, axis=0),
        color=Delta_color,
        label=r"$\Delta$",
        lw=3, alpha=0.8
    )
    ax1.fill_between(
        e_qs.solved_t_list,
        np.quantile(deltas_, 0.25, axis=0),
        np.quantile(deltas_, 0.75, axis=0),
        color=Delta_color, alpha=0.3
    )

    # ax1.set_ylabel(r"[GHz]")

    delta = (e_qs.t_list.max() - e_qs.t_list.min()) * 0.01
    ax1.set_xlim((e_qs.t_list.min() - delta, e_qs.t_list.max() + delta))

    # Panel 2

    ax2.plot(
        e_qs.solved_t_list[::foms_skip],
        np.mean(foms_, axis=0),
        lw=3,
        alpha=0.8,
        color="C2",
        zorder=2.1,
        label=r"$\mathcal{F}$",
    )
    ax2.fill_between(
        e_qs.solved_t_list[::foms_skip],
        np.quantile(foms_, 0.25, axis=0),
        np.quantile(foms_, 0.75, axis=0),
        color="C2", alpha=0.3
    )

    # ax2.plot(
    #     e_qs.solved_t_list,
    #     rho_00s,
    #     lw=2,
    #     alpha=0.8,
    #     color="C3",
    #     ls='--',
    #     label=r"$\rho_{00}$",
    # )
    # ax2.plot(
    #     e_qs.solved_t_list,
    #     rho_11s,
    #     lw=2,
    #     alpha=0.8,
    #     color="C4",
    #     ls='-.',
    #     label=r"$\rho_{11}$",
    # )
    # ax2.plot(
    #     e_qs.solved_t_list,
    #     off_diags,
    #     lw=2,
    #     alpha=0.8,
    #     color="C5",
    #     ls=':',
    #     label=r"$\rho_{01} + \rho_{10}$",
    # )

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

        e_qs = saver.load(BO_file, solve=False)

        fig = plt.figure(figsize=(3.5, 3))
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

        _plot_protocol_and_fidelity(
            ax1, ax2, e_qs, BO_file
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
        save_current_fig(f"_noisy_new_protocol_and_fidelity_{BO_file}")

        # ax2.legend(loc=9, framealpha=1)
        # save_current_fig(f"_legend_new_protocol_and_fidelity_{BO_file}")
        # raise RuntimeError()


if __name__ == '__main__':
    BO_FILES = [
        # f"12_BO_SHORT_STD_1D_std_",
        # f"12_BO_SHORT_STD_2D_std_",
        # f"12_BO_SHORT_STD_3D_std_",
        # f"12_BO_COMPARE_BO_WIDER_1D_alt_",
        f"12_BO_COMPARE_BO_2D_alt_",
        # f"12_BO_COMPARE_BO_3D_alt_",
        # "entanglement_entropy_ramp__8_2D_std",
        # "entanglement_entropy_ramp_2_8_2D_std",
        # "entanglement_entropy_ramp__12_2D_std",
        # "entanglement_entropy_ramp__16_2D_std",
        # "entanglement_entropy_ramp__8_2D_alt",
        # "entanglement_entropy_ramp__12_2D_alt",
        # "entanglement_entropy_ramp__16_2D_alt",
    ]
    plot_BO_geometries_and_GHZ()
