# Geometry exploration
from typing import Sequence, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes

import interaction_constants
from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.geometry.regular_lattice_2d import RegularLattice2D
from qubit_system.geometry.regular_lattice_3d import RegularLattice3D
from qubit_system.qubit_system_classes_quimb import StaticQubitSystem
from qubit_system.utils import states_quimb


class Geometry2D(BaseGeometry):
    def __init__(self, coordinates: np.ndarray):
        self.coordinates = coordinates


def plot_geometry(N: int, geometry: Geometry2D):
    plt.figure()

    plt.plot(geometry.coordinates[:, 0], geometry.coordinates[:, 1], 'x')
    for i, (x, y) in enumerate(geometry.coordinates):
        plt.text(x, y, i)

    plt.grid()
    plt.tight_layout()


def plot_detuning_energy_levels(s_qs: StaticQubitSystem, crossings: np.ndarray, ax: Axes,
                                highlighted_indices: Sequence[int] = (-1,)):
    if s_qs.Omega_zero_energies is None:
        s_qs.get_energies()

    crossings_range = crossings.max() - crossings.min()
    xlims = crossings_range * -0.1, crossings.max() * 1.1
    s_qs = StaticQubitSystem(
        s_qs.N, s_qs.V, s_qs.geometry, Omega=0, Delta=np.linspace(xlims[0], xlims[1], 20)
    )
    s_qs.get_energies()

    g = states_quimb.get_ground_states(1)[0]
    for i, state in enumerate(s_qs.states):
        is_highlight_state = any(state is s_qs.states[i] for i in highlighted_indices)
        is_ground_state = all((_state == g).all() for _state in state)

        color = 'g' if is_ground_state else 'r' if is_highlight_state else 'grey'
        linewidth = 5 if is_ground_state or is_highlight_state else 1
        z_order = 2 if is_ground_state or is_highlight_state else 1
        energies = s_qs.Omega_zero_energies[:, i]
        ax.plot(s_qs.Delta, energies,
                color=color,
                alpha=0.6,
                lw=linewidth, zorder=z_order,
                label=states_quimb.get_label_from_state(state),
                picker=3)

    def on_pick(event):
        line = event.artist
        print(f'Clicked on: {line.get_label()}')
    plt.gcf().canvas.mpl_connect('pick_event', on_pick)

    ax.grid()
    scaled_xaxis_ticker = ticker.EngFormatter(unit="Hz")
    scaled_yaxis_ticker = ticker.EngFormatter(unit="Hz")
    ax.xaxis.set_major_formatter(scaled_xaxis_ticker)
    ax.yaxis.set_major_formatter(scaled_yaxis_ticker)
    ax.locator_params(nbins=4)

    # plt.title(rf"Energy spectrum with $N = {self.N}$, $V = {self.V:0.2e}$, $\Omega = {self.Omega:0.2e}$")
    _m, _s = f"{s_qs.V:0.2e}".split('e')
    V_text = rf"{_m:s} \times 10^{{{int(_s):d}}}"
    plt.title(rf"Energy spectrum with $N = {s_qs.N}$, $V = {V_text:s}$ Hz")
    plt.xlabel(r"Detuning $\Delta$")
    plt.ylabel("Eigenenergy")

    plt.xlim(xlims)
    plt.tight_layout()


def calculate_ghz_crossings(s_qs: StaticQubitSystem, other_highlighted_labels: List[str] = ()):
    assert s_qs.N % 2 == 0, f"N has to be even, not {s_qs.N}"
    s_qs.get_energies()

    EEE_index = len(s_qs.states) - 1

    complementary_labels = [
        label.translate(str.maketrans({'e': 'g', 'g': 'e'}))
        for label in other_highlighted_labels
    ]
    other_highlighted_labels = other_highlighted_labels + [
        label for label in complementary_labels if label not in other_highlighted_labels
    ]
    other_highlighted_indices = []

    for i, state in enumerate(s_qs.states):
        label = states_quimb.get_label_from_state(state)
        if label in other_highlighted_labels:
            other_highlighted_indices.append(i)

    def find_root(x: np.ndarray, y: np.ndarray):
        """
        Finds crossing (where y equals 0), given that x, y is roughly linear.
        """
        if (y == 0).all():
            return np.nan
        _right_bound = (y < 0).argmax()
        _left_bound = _right_bound - 1
        crossing = y[_left_bound] / (y[_left_bound] - y[_right_bound]) \
                   * (x[_right_bound] - x[_left_bound]) + x[_left_bound]
        return crossing

    crossings = [find_root(s_qs.Delta, s_qs.Omega_zero_energies[:, i]) for i in range(len(s_qs.states))]
    # Crossing of GGG... with EEE...
    standard_GHZ_crossing = crossings[EEE_index]

    # Crossing of GGG... with EGEG...
    other_highlighted_crossings = [crossings[i] for i in other_highlighted_indices]

    crossings = np.array(crossings[1:])  # 1: removes first "crossing" of GGG (nan)
    unique_crossings, counts = np.unique(crossings, return_counts=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8))
    plot_detuning_energy_levels(s_qs, crossings, ax1, highlighted_indices=other_highlighted_indices + [-1])

    ax2.plot(unique_crossings, counts, 'x', alpha=0.4)
    ax2.grid()
    plt.show()
    pass


if __name__ == '__main__':
    N_RYD = 50
    C6 = interaction_constants.get_C6(N_RYD)

    LATTICE_SPACING = 1.5e-6

    print(f"C6: {C6:.3e}")
    characteristic_V = C6 / (LATTICE_SPACING ** 6)
    print(f"Characteristic V: {characteristic_V:.3e} Hz")

    N = 8

    geometry_and_labels = [
        (RegularLattice1D(), ['egegegeg']),
        (RegularLattice2D((4, 2)), ['eggeegge']),
        (RegularLattice3D((2, 2, 2)), ['eggegeeg']),
    ]
    i = 2
    geometry = geometry_and_labels[i][0]
    s_qs = StaticQubitSystem(
        N=N, V=characteristic_V, geometry=geometry,
        Omega=0,
        Delta=np.linspace(-characteristic_V * 5, characteristic_V * 5, 50),
    )

    geometry.plot()
    calculate_ghz_crossings(s_qs, other_highlighted_labels=geometry_and_labels[i][1])

    plt.show()
