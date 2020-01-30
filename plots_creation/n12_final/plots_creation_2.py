from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes

import interaction_constants
from plots_creation.n12_final.utils import save_current_fig
from qubit_system.geometry import RegularLattice
from qubit_system.qubit_systems.static_qubit_system import StaticQubitSystem
from qubit_system.utils import states

N = 12
LATTICE_SPACING = 1.5e-6

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)


def _plot_many_body_energy_spectrum_to_axis(ax: Axes, s_qs: StaticQubitSystem,
                                            highlight_states_by_label: Dict[str, str]):
    s_qs.get_energies()

    for i in range(len(s_qs.states)):
        label = states.get_label_from_state(s_qs.states[i])

        is_highlight_state = label in highlight_states_by_label
        color = highlight_states_by_label[label] if is_highlight_state else 'grey'
        linewidth = 3 if is_highlight_state else 1
        z_order = 2 if is_highlight_state else 1
        ax.plot(s_qs.Delta, s_qs.Omega_zero_energies[:, i], color=color, label=label, alpha=0.6, lw=linewidth,
                zorder=z_order)
    ax.grid()

    scaled_axis_ticker = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1e9))

    ax.xaxis.set_major_formatter(scaled_axis_ticker)
    ax.yaxis.set_major_formatter(scaled_axis_ticker)
    plt.locator_params(nbins=4)


def many_body_energy_spectrum(low_detuning: bool = False):
    if not low_detuning:
        detuning_limits = [-0.8e9, 2.8e9]
        y_lim = (-5e9, 10e9)
        name = f"energy_spectrum_{N}_1d2d3d"

    else:
        detuning_limits = [-0.1e9, 0.7e9]
        y_lim = (-2e9, 5e9)
        name = f"energy_spectrum_{N}_1d2d3d_low_detuning"

    fig_kwargs = {'figsize': (8, 3.2)}
    # fig_kwargs = {'figsize': (8, 6)}
    gridspec_kwargs = {
        'top': 0.95,
        'bottom': 0.2,
        'left': 0.08,
        'right': 0.98,
        'wspace': 0.05,
        'hspace': 0.05
    }
    fig, axs = plt.subplots(1, 3, sharey='all', sharex='all', **fig_kwargs, gridspec_kw=gridspec_kwargs)
    for i, shape in enumerate([
        (12,),
        (4, 3),
        (3, 2, 2)
    ]):
        s_qs = StaticQubitSystem(
            N, C6,
            geometry=RegularLattice(shape, LATTICE_SPACING),
            Omega=0, Delta=np.linspace(*detuning_limits, 10)
        )
        if len(shape) == 1:
            ghz_single_component = [True, False, True, False, True, False, True, False, True, False, True, False]
        elif len(shape) == 2:
            ghz_single_component = [True, False, True, False, True, False, True, False, True, False, True, False]
        elif len(shape) == 3:
            ghz_single_component = [True, False, False, True, False, True, True, False, True, False, False, True]

        ghz_label_1 = ''.join(
            ['e' if _ else 'g' for _ in ghz_single_component])
        ghz_label_2 = ''.join(
            ['g' if _ else 'e' for _ in ghz_single_component])
        ax = axs[i]
        _plot_many_body_energy_spectrum_to_axis(
            ax, s_qs,
            highlight_states_by_label={
                ''.join(['g'] * N): 'g',
                ''.join(['e'] * N): 'r',
                ghz_label_1: 'C0',
                ghz_label_2: 'C0',
            }
        )
        if not low_detuning:
            ax.text(0.98, 0.98, f"{len(shape)}D",
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes)

        ax.set_xlim(detuning_limits)
        ax.set_ylim(y_lim)
        if i == 0:
            ax.set_ylabel("[GHz]")
        ax.set_xlabel(r"$\Delta$ [GHz]")
    # fig.text(0.5, 0.04, xlabel, ha='center', va='center')
    # fig.text(0.02, 0.5, ylabel, ha='center', va='center', rotation='vertical')
    # plt.show()
    save_current_fig(name)


def many_body_energy_spectrum_omega_0():
    detuning_limits = [-0.8e9, 2.8e9]
    y_lim = (-5e9, 10e9)
    Omega = 1e9

    fig = plt.figure(figsize=(5, 4))
    ax = plt.gca()

    shape = (N,)
    s_qs = StaticQubitSystem(
        N, C6,
        geometry=RegularLattice(shape, LATTICE_SPACING),
        Omega=Omega, Delta=np.linspace(*detuning_limits, 500)
    )
    s_qs.get_energies()

    for i in range(len(s_qs.states)):
        color = 'grey'
        linewidth = 1
        z_order = 1
        ax.plot(s_qs.Delta, s_qs.Omega_non_zero_energies[:, i], color=color, alpha=0.6, lw=linewidth,
                zorder=z_order)
    ax.grid()
    scaled_xaxis_ticker = ticker.EngFormatter(unit="Hz")
    scaled_yaxis_ticker = ticker.EngFormatter(unit="Hz")
    ax.xaxis.set_major_formatter(scaled_xaxis_ticker)
    ax.yaxis.set_major_formatter(scaled_yaxis_ticker)
    plt.locator_params(nbins=4)

    # ax.set_xlabel(f"{len(shape)}D")
    ax.text(0.95, 0.95, f"$\Omega = {Omega / 1e9}$ GHz", horizontalalignment='right',
            verticalalignment='center', transform=ax.transAxes)

    ax.set_xlim(detuning_limits)
    ax.set_ylim(y_lim)

    xlabel = r"Detuning $\Delta$"
    ylabel = "Eigenenergy"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    save_current_fig(f"energy_spectrum_{N}_1d_omega_nonzero")


if __name__ == '__main__':
    # many_body_energy_spectrum()
    many_body_energy_spectrum(low_detuning=True)
    # many_body_energy_spectrum_omega_0()
