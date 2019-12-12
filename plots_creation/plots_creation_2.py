import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes
from scipy import interpolate
import quimb as q

import interaction_constants
from optimised_protocols import saver
from plots_creation.utils import save_current_fig
from qubit_system.geometry import BaseGeometry
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem
from qubit_system.utils.ghz_states import CustomGHZState

N = 8
LATTICE_SPACING = 1.5e-6

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)


# def _get_e_qs(D: int, GHZ: str, geometry: BaseGeometry,
#               Omega: np.ndarray, Delta: np.ndarray,
#               t_list: np.ndarray):
#     N = 8
#     assert GHZ == "std" or GHZ == "alt", f"GHZ has to be std or alt, not {GHZ}"
#     ghz_component = [True, True, True, True, True, True, True, True] if GHZ == "std" else None
#     if GHZ == "alt":
#         if D == 1:
#             ghz_component = [True, False, True, False, True, False, True, False]
#         elif D == 2:
#             ghz_component = [True, False, False, True, True, False, False, True]
#         elif D == 3:
#             ghz_component = [True, False, False, True, False, True, True, False]
#         else:
#             raise ValueError(f"Unknown D: {D}")
#
#     ghz_state = CustomGHZState(N, ghz_component)
#
#     t = 2e-6
#     interpolation_timesteps = 3000
#     t_list = np.linspace(0, t, interpolation_timesteps + 1)
#
#     e_qs = EvolvingQubitSystem(
#         N, C6, geometry,
#         Omega, Delta,
#         t_list,
#         ghz_state=ghz_state
#     )
#     e_qs.solve()
#
#     ghz_fidelity = e_qs.get_fidelity_with("ghz")
#
#     print(f"{D}D {GHZ} fidelity: {ghz_fidelity:.5f}")
#     return e_qs


def _plot_protocol_and_fidelity(ax1: Axes, ax2: Axes, e_qs: EvolvingQubitSystem, with_antisymmetric_ghz: bool,
                                first: bool, last: bool):
    ax1.xaxis.set_major_formatter(ticker.EngFormatter('s'))
    Omega = interpolate.interp1d(e_qs.t_list, np.hstack((e_qs.Omega, e_qs.Omega[-1])), kind="previous",
                                 fill_value="extrapolate")
    Delta = interpolate.interp1d(e_qs.t_list, np.hstack((e_qs.Delta, e_qs.Delta[-1])), kind="previous",
                                 fill_value="extrapolate")

    Omega_color = "C0"
    Delta_color = "C3"

    ax1.plot(e_qs.solved_t_list, [Omega(t) / 1e9 for t in e_qs.solved_t_list], color=Omega_color, lw=3, alpha=0.8)
    ax1.locator_params(nbins=3, axis='x')

    Delta_ax = ax1.twinx()
    Delta_ax.plot(e_qs.solved_t_list, [Delta(t) / 1e9 for t in e_qs.solved_t_list], color=Delta_color, lw=3, alpha=0.8)

    # ax1.yaxis.set_major_formatter(ticker.EngFormatter('Hz'))
    ax1.locator_params(nbins=4, axis='y')
    if first:
        ax1.set_ylabel(r"$\Omega (t)$ [GHz]", color=Omega_color)
    ax1.yaxis.label.set_color(Omega_color)
    ax1.tick_params(axis='y', labelcolor=Omega_color)
    # Delta_ax.yaxis.set_major_formatter(ticker.EngFormatter('Hz'))
    Delta_ax.locator_params(nbins=4, axis='y')
    if last:
        Delta_ax.set_ylabel(r"$\Delta (t)$ [GHz]", color=Delta_color)
    Delta_ax.yaxis.label.set_color(Delta_color)
    Delta_ax.tick_params(axis='y', labelcolor=Delta_color)

    delta = (e_qs.t_list.max() - e_qs.t_list.min()) * 0.01
    ax1.set_xlim((e_qs.t_list.min() - delta, e_qs.t_list.max() + delta))

    # Panel 2
    labelled_states = [(e_qs.ghz_state.get_state_tensor(), r"$\mathcal{F}_S$")]
    if with_antisymmetric_ghz:
        labelled_states.append(
            (e_qs.ghz_state.get_state_tensor(symmetric=False), r"$\mathcal{F}_A$"))

    colors = {
        0: "C2",
        1: "C1"
    }
    for i, (_state, _label) in enumerate(labelled_states):
        ax2.plot(
            e_qs.solved_t_list,
            [q.fidelity(_state, _instantaneous_state)
             for _instantaneous_state in e_qs.solved_states],
            label=_label,
            lw=1,
            alpha=0.8,
            color=colors[i],
        )
    if first:
        ax2.set_ylabel(r"Fidelity")
    if with_antisymmetric_ghz:
        ax2.legend(loc=2)

    ax2.set_ylim((-0.1, 1.1))
    ax2.yaxis.set_ticks([0, 0.5, 1])
    ax2.grid()


def plot_ml_methods_protocol_and_fidelity():
    manual = saver.load("MANUAL_1D_STD_8_0.989_2019-11-26T16-43-02.024761.json")
    chosen_protocols = {
        'MANUAL': manual,
        'GRAPE': saver.load("GRAPE_1D_STD_8_0.167_2019-11-26T13-33-57.433389.json"),
        'Krotov': saver.load("KROTOV_1D_STD_8_0.997_2019-11-26T17-55-48.632780.json"),
        'CRAB': saver.load("CRAB_1D_STD_8_0.027_2019-11-26T15-03-46.544170.json"),
        'RL': saver.load("RL_1D_STD_8_0.318_2019-11-27T12-34-22.925263.json"),
        'BO': saver.load("BO_1D_STD_8_0.923_2019-11-26T17-51-44.508830.json"),
    }
    fig, (axs) = plt.subplots(2, len(chosen_protocols), sharex='col', figsize=(16, 5),
                              gridspec_kw={'wspace': 0.4, 'hspace': 0.05, 'left': 0.05, 'right': 0.95})
    for i, (method, e_qs) in enumerate(chosen_protocols.items()):
        ax1 = axs[0, i]
        ax2 = axs[1, i]
        _plot_protocol_and_fidelity(
            ax1, ax2, e_qs, with_antisymmetric_ghz=False,
            first=i == 0, last=i == len(chosen_protocols) - 1
        )
        ax1.set_title(method)

    plt.xlabel('Time')
    plt.tight_layout()
    save_current_fig("ml_methods_protocol_and_fidelity")


def plot_BO_geometries_and_GHZ():
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(5, 3, wspace=0.3, hspace=0.05, height_ratios=[1, 1, 0.7, 1, 1],
                          top=0.95, bottom=0.05, left=0.05, right=0.95)
    # fig, (axs) = plt.subplots(4, 3, sharex='col', figsize=(16, 6))

    e_qs = None
    for col, shape in enumerate([(8,), (4, 2), (2, 2, 2)]):
        D = len(shape)
        for row, ghz in enumerate(["std", "alt"]):
            BO_file = f"BO_COMPARE_BO_{D}D_{ghz}_8"

            # if e_qs is None:
            #     e_qs = saver.load(BO_file)
            e_qs = saver.load(BO_file)

            row_ = row * 3
            ax1 = fig.add_subplot(gs[row_, col])
            ax2 = fig.add_subplot(gs[row_ + 1, col], sharex=ax1)
            # ax1 = axs[row_, col]
            # ax2 = axs[row_ + 1, col]
            _plot_protocol_and_fidelity(
                ax1, ax2, e_qs, with_antisymmetric_ghz=True,
                first=col == 0, last=col == 2
            )
            ghz_ket = r"\ghzstd" if ghz == "std" else r"\ghzalt"
            ax1.set_title(f"{len(shape)}D,  " + r"$\qquad \ket{\psi_{\mathrm{target}}} = " + ghz_ket + "$")
            # ax1.text(0.95, 0.95, f"{len(shape)}D\n" + r"$\ghzstd$", horizontalalignment='center',
            #          verticalalignment='center', transform=ax1.transAxes)
            ax1.get_xaxis().set_visible(False)
    save_current_fig("bo_geometries_ghz_protocol_and_fidelity")


if __name__ == '__main__':
    plot_ml_methods_protocol_and_fidelity()
    # plot_BO_geometries_and_GHZ()
