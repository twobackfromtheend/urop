import matplotlib.pyplot as plt
import numpy as np
import quimb as q
from matplotlib import ticker
from matplotlib.axes import Axes
from scipy import interpolate

import interaction_constants
from optimised_protocols import saver
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem
from plots_creation.n12_final.utils import save_current_fig
from qubit_system.utils.ghz_states import StandardGHZState, CustomGHZState

N = 12
LATTICE_SPACING = 1.5e-6

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)


def _plot_protocol_and_fidelity(ax1: Axes, ax2: Axes, e_qs: EvolvingQubitSystem,
                                is_alt: bool,
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

    ghz_state = e_qs.ghz_state
    component_1, component_2 = ghz_state._get_components()
    # Panel 2
    if not is_alt:
        labelled_states = [(component_1, r"$\mathcal{F}_{GG}$"), (component_2, r"$\mathcal{F}_{EE}$")]
    else:
        labelled_states = [(component_1, r"$\mathcal{F}_{EG}$"), (component_2, r"$\mathcal{F}_{GE}$")]

    colors = {
        0: "tab:green",
        1: "tab:red"
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
    ax2.legend()
    if not first_row:
        ax2.set_xlabel(r"[$\upmu$s]")

    ax2.set_ylim((-0.1, 1.1))
    ax2.yaxis.set_ticks([0, 0.5, 1])
    ax2.grid()


def plot_BO_geometries_and_GHZ():
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(5, 3, wspace=0.3, hspace=0.05, height_ratios=[1, 1, 0.5, 1, 1],
                          top=0.93, bottom=0.09, left=0.08, right=0.98)
    for col, D in enumerate([1, 2, 3]):
        for row, ghz in enumerate(["std", "alt"]):
            if D == 1:
                BO_file = f"12_BO_COMPARE_BO_WIDER_{D}D_{ghz}_"
            else:
                BO_file = f"12_BO_COMPARE_BO_{D}D_{ghz}_"
            # if e_qs is None:
            #     e_qs = saver.load(BO_file)
            e_qs = saver.load(BO_file)

            row_ = row * 3
            ax1 = fig.add_subplot(gs[row_, col])
            ax2 = fig.add_subplot(gs[row_ + 1, col], sharex=ax1)
            _plot_protocol_and_fidelity(
                ax1, ax2, e_qs,
                is_alt=ghz == "alt",
                first=col == 0, last=col == 2,
                first_row=row_ == 0
            )
            ghz_ket = r"\ghzstd" if ghz == "std" else r"\ghzalt"
            ax1.set_title(f"{D}D,  " + r"$\quad \ket{\psi_{\mathrm{target}}} = " + ghz_ket + "$")
            ax1.get_xaxis().set_visible(False)
    save_current_fig("target_state_bo_geometries_ghz_protocol_and_fidelity")


def zoomed_plots_near_final():
    class NullAxis:
        def __init__(self):
            self.xaxis = self
            self.yaxis = self

        def plot(self, *args, **kwargs):
            pass

        def set_major_formatter(self, *args, **kwargs):
            pass

        def locator_params(self, *args, **kwargs):
            pass

        def set_ylabel(self, *args, **kwargs):
            pass

        def set_xlim(self, *args, **kwargs):
            pass

    for col, shape in enumerate([(12,), (4, 3), (3, 2, 2)]):
        D = len(shape)
        for row, ghz in enumerate(["std"]):
            BO_file = f"12_BO_COMPARE_BO_{D}D_{ghz}_"
            fig = plt.figure(figsize=(4, 3.5))
            ax1 = NullAxis()
            ax2 = plt.gca()

            e_qs = saver.load(BO_file)

            _plot_protocol_and_fidelity(
                ax1, ax2, e_qs, with_antisymmetric_ghz=True,
                first=True, last=col == 2,
                first_row=True
            )
            ax2.set_xlim([0.98e-6, 1.001e-6])
            ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1e6)))
            # plt.show()
            save_current_fig(f"ee_gg_zoomed_{BO_file}")


if __name__ == '__main__':
    plot_BO_geometries_and_GHZ()
    # zoomed_plots_near_final()
