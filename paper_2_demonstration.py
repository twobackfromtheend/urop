from collections import defaultdict

from matplotlib import ticker
import matplotlib.pyplot as plt
from scipy import constants
import numpy as np
from qutip import *
from hamiltonian import get_hamiltonian
from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.geometry.regular_lattice_2d import RegularLattice2D
from qubit_system.qubit_system_classes import StaticQubitSystem
from states import get_states, is_excited, get_label_from_state
import tqdm


def plot_n_4():
    N = 4
    a = 532e-9

    # C6 = 2 * constants.pi * 0.4125e6 * (8 * a) ** 6  # [MHz m^6]
    C6 = 1.625e-60  # [J m^6]

    L = 8 * a
    V_L = C6 / (L ** 6) / constants.hbar  # Joules / hbar = Hz
    V = C6 / constants.hbar  # [Hz / m6]
    # "/ constants.hbar" due to hamiltonian definition (H / hbar as input)

    s_qs = StaticQubitSystem(
        N=N, V=V, geometry=RegularLattice2D((2, 2), spacing=L),
        Omega=0,
        Delta=np.linspace(-0.5 * V_L, 2.5 * V_L, 3)
    )

    s_qs.plot_detuning_energy_levels(plot_state_names=False)

    ax = plt.gca()
    scaled_axis_ticker = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / V_L))
    ax.xaxis.set_ticks(np.arange(-0.5, 2.51, 0.5) * V_L)
    ax.yaxis.set_ticks(np.arange(-4, 61, 2) * V_L)
    ax.xaxis.set_major_formatter(scaled_axis_ticker)
    ax.yaxis.set_major_formatter(scaled_axis_ticker)
    ax.set_xlim(-0.5 * V_L, 2.5 * V_L)
    ax.set_ylim(-5 * V_L, 6 * V_L)
    plt.show()


def plot_n_8():
    N = 8
    a = 532e-9

    # C6 = 2 * constants.pi * 0.4125e6 * (8 * a) ** 6  # [MHz m^6]
    C6 = 1.625e-60  # [J m^6]

    L = (N - 1) * a
    V_L = C6 / (L ** 6) / constants.hbar  # Joules / hbar
    V = C6 / constants.hbar
    # "/ constants.hbar" due to hamiltonian definition (H / hbar as input)

    s_qs = StaticQubitSystem(
        N=N, V=V, geometry=RegularLattice1D(spacing=a),
        Omega=0,
        Delta=np.linspace(-3e4 * V_L, 23e4 * V_L, 3)
    )

    s_qs.plot_detuning_energy_levels(plot_state_names=False)

    ax = plt.gca()
    ax.xaxis.set_ticks(np.arange(0, 21e4, 5e4) * V_L)
    ax.yaxis.set_ticks(np.arange(-10e4, 11e4, 5e4) * V_L)
    ax.set_xlim(-3e4 * V_L, 23e4 * V_L)
    ax.set_ylim(-10 * V_L * 7 ** 6, 10 * V_L * 7 ** 6)  # WHY IS THERE A 7^6 TERM?!
    # ax.set_ylim(-10 * V_L, 10 * V_L)

    scaled_xaxis_ticker = ticker.FuncFormatter(lambda x, pos: '{0:.1e}'.format(x / V_L))
    scaled_yaxis_ticker = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / V_L))
    ax.xaxis.set_major_formatter(scaled_xaxis_ticker)
    ax.yaxis.set_major_formatter(scaled_yaxis_ticker)
    plt.show()

    -.3, 1.4


plot_n_4()
