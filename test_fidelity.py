import matplotlib.pyplot as plt
import numpy as np
import quimb as q

from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem
from statistics_plot import get_e_qs


def test_figure_of_merit(e_qs: EvolvingQubitSystem):
    ax = plt.gca()
    e_qs.plot_ghz_states_overlaps(ax, False, False)

    states = e_qs.solved_states

    ghz_ket = e_qs.ghz_state.get_state_tensor()
    ghz_bra = ghz_ket.H

    last_state_density_matrix = np.array(states[-1] @ states[-1].H)
    last_state_fidelity = q.fidelity(ghz_ket, states[-1])

    ghz_indices = ghz_ket.nonzero()[0]

    p0 = last_state_density_matrix[ghz_indices[0], ghz_indices[0]]
    p1 = last_state_density_matrix[ghz_indices[1], ghz_indices[1]]

    c0 = last_state_density_matrix[ghz_indices[0], 255 - ghz_indices[0]]
    c1 = last_state_density_matrix[ghz_indices[1], 255 - ghz_indices[1]]

    new_last_state_fidelity = (p0 + p1 + c0 + c1) / 2
    print(f"fidelity: {last_state_fidelity:.3f}, {new_last_state_fidelity:.3f}")

    fidelities1 = []
    fidelities2 = []
    for state in states:
        fidelity1 = q.fidelity(ghz_ket, state)
        fidelities1.append(fidelity1)

        density_matrix = state @ state.H
        # Diagonal elements
        p0 = density_matrix[ghz_indices[0], ghz_indices[0]]
        p1 = density_matrix[ghz_indices[1], ghz_indices[1]]
        # Off-diagonal elements
        c0 = density_matrix[ghz_indices[0], 255 - ghz_indices[0]]
        c1 = density_matrix[ghz_indices[1], 255 - ghz_indices[1]]
        fidelity2 = (p0 + p1 + c0 + c1) / 2
        fidelities2.append(fidelity2.real)

    print(f"Fidelities equal? {np.isclose(fidelities1, fidelities2).all()}")
    ax.plot(e_qs.solved_t_list, fidelities2, 'C1--', alpha=0.3)

    plt.show()


setups = [
    (1, "std"),
    (1, "alt"),
    (2, "std"),
    (2, "alt"),
]
for D, ghz in setups:
    e_qs = get_e_qs(D, ghz)
    name = f"{D}_{ghz}_{e_qs.get_fidelity_with('ghz'):.5f}"
    test_figure_of_merit(e_qs)
