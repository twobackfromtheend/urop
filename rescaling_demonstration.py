import numpy as np
from matplotlib import ticker, pyplot as plt

import paper_data

from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.geometry.regular_lattice_2d import RegularLattice2D
from qubit_system.qubit_system_classes import StaticQubitSystem, EvolvingQubitSystem
from qubit_system.utils.interpolation import get_hamiltonian_coeff_linear_interpolation
from qubit_system.utils.states import get_ghz_state, GHZStateType


def rescale_evolving_qubit_system():
    t = 1.1e-6
    N = 4
    e_qs = EvolvingQubitSystem(
        N=N, V=paper_data.V, geometry=RegularLattice1D(),
        Omega=paper_data.get_hamiltonian_coeff_fn(paper_data.Omega, N),
        Delta=paper_data.get_hamiltonian_coeff_fn(paper_data.Delta, N),
        t_list=np.linspace(0, t, 100),
        ghz_state=get_ghz_state(N, _type=GHZStateType.ALTERNATING)
    )
    e_qs.solve()
    # e_qs.plot()

    print(e_qs.get_fidelity_with("ghz"))

    max_Omega = 50e6

    e_qs = EvolvingQubitSystem(
        N=N, V=paper_data.V / max_Omega, geometry=RegularLattice1D(),
        Omega=paper_data.get_hamiltonian_coeff_fn(
            {k: {_k * max_Omega: _v / max_Omega for _k, _v in v.items()} for k, v in paper_data.Omega.items()}, N),
        Delta=paper_data.get_hamiltonian_coeff_fn(
            {k: {_k * max_Omega: _v / max_Omega for _k, _v in v.items()} for k, v in paper_data.Delta.items()}, N),
        t_list=np.linspace(0, t * max_Omega, 100),
        ghz_state=get_ghz_state(N, _type=GHZStateType.ALTERNATING)
    )

    e_qs.solve()
    e_qs.plot()

    print(e_qs.get_fidelity_with("ghz"))


def rescale_static_qubit_system():
    N = 4
    Delta = np.linspace(-15e6, 20e7, 16)
    Omega = 3e7
    s_qs_1 = StaticQubitSystem(
        N=N, V=paper_data.V, geometry=RegularLattice1D(),
        Omega=Omega,
        Delta=Delta,
    )
    s_qs_1.plot_detuning_energy_levels(plot_state_names=True, show=True)

    # Rescaled
    max_Omega = Omega

    s_qs_2 = StaticQubitSystem(
        N=N, V=paper_data.V / max_Omega, geometry=RegularLattice1D(),
        Omega=Omega / max_Omega,
        Delta=Delta / max_Omega,
    )
    s_qs_2.plot_detuning_energy_levels(plot_state_names=True)

    ax = plt.gca()

    ax.set_xticks(np.arange(0, 2.1e8, 0.5e8) / max_Omega)
    ax.set_yticks(np.arange(-4e8, 5e8, 2e8) / max_Omega)

    scaled_xaxis_ticker = ticker.FuncFormatter(lambda x, pos: '{0:.1e}'.format(x * max_Omega))
    scaled_yaxis_ticker = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * max_Omega))
    ax.xaxis.set_major_formatter(scaled_xaxis_ticker)
    ax.yaxis.set_major_formatter(scaled_yaxis_ticker)
    plt.tight_layout()

    plt.show()


rescale_static_qubit_system()
