from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes

import interaction_constants
from qubit_system.geometry import RegularLattice
from qubit_system.qubit_systems.static_qubit_system import StaticQubitSystem
from qubit_system.utils import states
from qutip_job_handlers.qutip_bo_paper_plots_utils import _get_ghz_single_component_from_dimension, save_current_fig

N = 9
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
    scaled_xaxis_ticker = ticker.EngFormatter(unit="Hz")
    scaled_yaxis_ticker = ticker.EngFormatter(unit="Hz")
    ax.xaxis.set_major_formatter(scaled_xaxis_ticker)
    ax.yaxis.set_major_formatter(scaled_yaxis_ticker)
    plt.locator_params(nbins=4)


def many_body_energy_spectrum():
    detuning_limits = [-0.8e9, 3.2e9]
    y_lim = (-5e9, 10e9)

    for N, shape in [
        (9, (9,)),
        (9, (3, 3)),
        (8, (2, 2, 2)),
    ]:
        name = f"energy_spectrum_{N}_{len(shape)}d"

        fig = plt.figure(figsize=(5, 4))
        ax = plt.gca()

        s_qs = StaticQubitSystem(
            N, C6,
            geometry=RegularLattice(shape, LATTICE_SPACING),
            Omega=0, Delta=np.linspace(*detuning_limits, 10)
        )
        # ghz_label_1 = ''.join(
        #     ['e' if _ else 'g' for _ in _get_ghz_single_component_from_dimension(len(shape), alt=True)])
        # ghz_label_2 = ''.join(
        #     ['g' if _ else 'e' for _ in _get_ghz_single_component_from_dimension(len(shape), alt=True)])
        _plot_many_body_energy_spectrum_to_axis(
            ax, s_qs,
            highlight_states_by_label={
                ''.join(['g'] * N): 'g',
                ''.join(['e'] * N): 'r',
                # ghz_label_1: 'C0',
                # ghz_label_2: 'C0',
            }
        )

        ax.set_xlim(detuning_limits)
        ax.set_ylim(y_lim)

        xlabel = r"Detuning $\Delta$"
        ylabel = "Eigenenergy"

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        save_current_fig(name)


def many_body_energy_spectrum_omega_0():
    detuning_limits = [-0.8e9, 3.2e9]
    y_lim = (-5e9, 10e9)
    # Omega = 1e9
    Omega = 500e6

    for N, shape in [
        (9, (9,)),
        (9, (3, 3)),
        (8, (2, 2, 2)),
    ]:
        name = f"energy_spectrum_{N}_{len(shape)}d"
        fig = plt.figure(figsize=(5, 4))
        ax = plt.gca()

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

        # ax.text(0.95, 0.95, f"$\Omega = {Omega / 1e6}$ MHz", horizontalalignment='right',
        #         verticalalignment='center', transform=ax.transAxes)

        ax.set_xlim(detuning_limits)
        ax.set_ylim(y_lim)

        xlabel = r"Detuning $\Delta$"
        ylabel = "Eigenenergy"

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        save_current_fig(f"{name}_omega_nonzero")


if __name__ == '__main__':
    many_body_energy_spectrum()
    # many_body_energy_spectrum(low_detuning=True)
    # many_body_energy_spectrum_omega_0()
