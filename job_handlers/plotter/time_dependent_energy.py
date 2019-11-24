from typing import List

import matplotlib.pyplot as plt
import numpy as np
import quimb as q
from matplotlib import ticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from tqdm import tqdm

from job_handlers.hamiltonian import SpinHamiltonian, QType
from job_handlers.solver import solve_with_protocol
from qubit_system.geometry import BaseGeometry
from qubit_system.utils.ghz_states import BaseGHZState


def plot_time_dependent_eigenenergies(spin_ham: SpinHamiltonian,
                                      V: float, geometry: BaseGeometry, t_list: np.ndarray,
                                      Omega: np.ndarray, Delta: np.ndarray,
                                      states: q.qarray, ghz_state: BaseGHZState):
    ghz_state_tensor = q.qu(ghz_state.get_state_tensor(), sparse=False)

    eigenenergies = []
    eigenstate_populations = []
    ghz_fidelities = []
    plot_t_list = []
    for i in tqdm(range(len(Omega))):
        if i % 5 != 0:
            continue
        plot_t_list.append(t_list[i])
        _Omega = Omega[i]
        _Delta = Delta[i]
        state = states[i]

        hamiltonian = spin_ham.get_hamiltonian(V, geometry, _Omega, _Delta)

        dense_hamiltonian = q.qu(hamiltonian, sparse=False)
        instantaneous_eigenenergies, instantaneous_eigenstates = q.eigh(dense_hamiltonian)
        instantaneous_eigenstates = instantaneous_eigenstates.T

        _eigenenergies = []
        _eigenstate_populations = []
        _ghz_fidelities = []
        for eigenenergy, eigenstate in zip(instantaneous_eigenenergies, instantaneous_eigenstates):
            eigenenergy = eigenenergy - instantaneous_eigenenergies.min()
            if eigenenergy > 1e9:
                continue
            _eigenenergies.append(eigenenergy)
            eigenstate_population = q.fidelity(state, eigenstate)
            _eigenstate_populations.append(eigenstate_population)
            _eigenstate_ghz_fidelity = q.fidelity(ghz_state_tensor, eigenstate)
            _ghz_fidelities.append(_eigenstate_ghz_fidelity)
        eigenenergies.append(_eigenenergies)
        eigenstate_populations.append(_eigenstate_populations)
        ghz_fidelities.append(_ghz_fidelities)

    eigenenergies = np.array(eigenenergies).T
    eigenstate_populations = np.array(eigenstate_populations).T
    ghz_fidelities = np.array(ghz_fidelities).T

    cmap = plt.cm.get_cmap('viridis')
    norm = LogNorm(1e-4, 1)
    for i, _eigenenergies  in enumerate(eigenenergies):
        _eigenstate_populations = eigenstate_populations[i]
        _ghz_fidelities = ghz_fidelities[i]
        # color = cmap(np.log10(np.clip(_eigenstate_populations, 1e-4, 1)))
        plt.scatter(plot_t_list, _eigenenergies,
                    s=3,
                    # c=np.clip(_eigenstate_populations, 1e-4, 1),
                    c=np.clip(_ghz_fidelities, 1e-4, 1),
                    cmap=cmap, norm=norm,
                    # c='k',
                    edgecolors='none',
                    alpha=0.4)

    ax = plt.gca()
    delta = (t_list.max() - t_list.min()) * 0.01
    ax.set_xlim((t_list.min() - delta, t_list.max() + delta))

    ax.xaxis.set_major_formatter(ticker.EngFormatter('s'))
    plt.xlabel('Time')
    ax.yaxis.set_major_formatter(ticker.EngFormatter('Hz'))
    plt.ylabel('Eigenenergies')
    ax.locator_params(nbins=4, axis='y')
    ax.locator_params(nbins=5, axis='x')

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, ticks=[1e-4, 1e-3, 1e-2, 1e-1, 1], extend='min')
    # cbar.ax.set_ylabel(r"Eigenstate population")
    cbar.ax.set_ylabel(r"GHZ fidelity")

    plt.tight_layout()
    plt.grid()
    # plt.show()


if __name__ == '__main__':
    from job_handlers.plotter.optimised_protocols import get_optimised_protocol
    from job_handlers.timer import timer

    setups = [
        # (1, "std"),
        (1, "alt"),
        # (2, "std"),
        (2, "alt"),
    ]
    for D, GHZ in setups:
        N, V, geometry, t_list, psi_0, ghz_state, Omega, Delta = get_optimised_protocol(D, GHZ)

        with timer(f"Loading SpinHam (N={N})"):
            spin_ham = SpinHamiltonian.load(N)

        with timer("Solving with protocol"):
            states = solve_with_protocol(
                spin_ham, V=V, geometry=geometry, t_list=t_list, psi_0=psi_0,
                Omega=Omega, Delta=Delta
            )
            # TODO: Fix, now only returns final state.

        plt.figure(figsize=(10, 7))
        plot_time_dependent_eigenenergies(spin_ham,
                                          V=V, geometry=geometry, t_list=t_list, Omega=Omega, Delta=Delta,
                                          states=states,
                                          ghz_state=ghz_state)
        plt.savefig(f"plots/time_dependent_eigenenergies_{D}_{GHZ}_gap_limited_coloured_ghz.png", dpi=300)
