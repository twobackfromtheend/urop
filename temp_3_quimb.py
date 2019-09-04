import math
import time
from collections import defaultdict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import quimb as q
from bitarray import bitarray
from matplotlib import ticker
from tqdm import tqdm

import paper_data
from paper_data.interpolation import get_hamiltonian_coeff_fn
from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.qubit_system_classes_quimb import EvolvingQubitSystem
from qubit_system.utils.ghz_states_quimb import AlternatingGHZState

plt.rcParams['savefig.dpi'] = 600
# plt.plot([0, 1.1], [-18, 18])
# plt.grid()
# plt.tight_layout()
# plt.show()
from qubit_system.utils.states_quimb import get_states, get_label_from_state, get_product_basis_states_index

t = 1.1e-6
N = 20
ghz_state = AlternatingGHZState(N)
num_ts = 150
Omega_func = get_hamiltonian_coeff_fn(paper_data.Omega, N)
Delta_func = get_hamiltonian_coeff_fn(paper_data.Delta, N)
Omega = np.array([Omega_func(_t) for _t in np.linspace(0, t, num_ts)])[:-1]
Delta = np.array([Delta_func(_t) for _t in np.linspace(0, t, num_ts)])[:-1]
e_qs = EvolvingQubitSystem(
    N=N, V=paper_data.V, geometry=RegularLattice1D(),
    Omega=Omega,
    Delta=Delta,
    t_list=np.linspace(0, t, num_ts),
    ghz_state=ghz_state
)


def get_hamiltonian_variables_with_edge_fields(e_qs: EvolvingQubitSystem) -> Tuple[q.qarray, q.qarray, q.qarray]:
    sx = q.pauli("X", sparse=True)
    sz = q.pauli("Z", sparse=True)
    qnum = (sz + q.identity(2, sparse=True)) / 2
    dims = [2] * e_qs.N

    # noinspection PyTypeChecker
    time_independent_terms: q.qarray = 0
    # noinspection PyTypeChecker
    Omega_coeff_terms: q.qarray = 0
    # noinspection PyTypeChecker
    Delta_coeff_terms: q.qarray = 0

    for i in range(e_qs.N):
        Omega_coeff_terms += q.ikron(sx, dims=dims, inds=i, sparse=True) / 2
        n_i = q.ikron(qnum, dims=dims, inds=i, sparse=True)
        Delta_coeff_terms -= n_i

        if e_qs.N <= 8:
            if i == 0 or i == e_qs.N - 1:
                time_independent_terms += n_i * 4.5e6 * 2 * np.pi
        elif e_qs.N > 8:
            if i == 0 or i == e_qs.N - 1:
                time_independent_terms += n_i * 6e6 * 2 * np.pi
            elif i == 3 or i == e_qs.N - 4:
                time_independent_terms += n_i * 1.5e6 * 2 * np.pi

        for j in range(i):
            n_j = q.ikron(qnum, dims=dims, inds=j, sparse=True)

            time_independent_terms += e_qs.V / e_qs.geometry.get_distance(i, j) ** 6 * n_i * n_j

    return (
        time_independent_terms,
        Omega_coeff_terms,
        Delta_coeff_terms
    )


hamiltonian_variables = get_hamiltonian_variables_with_edge_fields(e_qs)


def get_hamiltonian_with_edge_fields(Omega: float, Delta: float):
    return hamiltonian_variables[0] + hamiltonian_variables[1] * Omega + hamiltonian_variables[2] * Delta


start_time = time.time()


# noinspection PyTypeChecker
def solve(e_qs: EvolvingQubitSystem) -> List[q.qarray]:
    dt = e_qs.t_list[1]

    e_qs.solved_states = [e_qs.psi_0]
    e_qs.solved_t_list = [0]

    latest_state = state = e_qs.psi_0
    latest_time = 0
    for i in tqdm(range(len(e_qs.Omega))):
        Omega = e_qs.Omega[i]
        Delta = e_qs.Delta[i]
        evo = q.Evolution(
            latest_state,
            get_hamiltonian_with_edge_fields(Omega, Delta),
            # method="expm",
            # progbar=True,
        )
        solve_points = np.linspace(0, dt, e_qs.solve_points_per_timestep + 1)[1:]  # Take away t=0 as a solve point
        for state in evo.at_times(solve_points):
            e_qs.solved_states.append(state)
        e_qs.solved_t_list += (solve_points + latest_time).tolist()
        latest_state = state
        latest_time = e_qs.solved_t_list[-1]

    return e_qs.solved_states


solved_states = solve(e_qs)

print(f"solved in {time.time() - start_time:.3f}s")


def calculate_subsystem_entropy(state: q.qarray):
    return q.entropy_subsys(state, [2] * e_qs.N, np.arange(e_qs.N / 2))


start_time = time.time()
subsystem_entropy = [calculate_subsystem_entropy(state_) for state_ in e_qs.solved_states]
print(f"calculated entropies in {time.time() - start_time:.3f}s")
print("plotting.")

fig, axs = plt.subplots(3, 1, sharex='all', figsize=(10, 8))
ax0, ax1, ax2 = axs
ax0.xaxis.set_major_formatter(ticker.EngFormatter('s'))
plt.xlabel('Time')

e_qs.plot_Omega_and_Delta(ax0)

ghz_state_tensor = ghz_state.get_state_tensor(True)

ax1.plot(
    e_qs.t_list,
    [q.fidelity(ghz_state_tensor, _state) for _state in solved_states],
    label=r"$\psi_{\mathrm{GHZ}}^{\mathrm{s}}$",
    lw=1,
    alpha=0.8
)
ax1.set_ylabel("Fidelity")
ax1.set_title("Fidelity with GHZ states")
ax1.set_ylim((-0.1, 1.1))
ax1.yaxis.set_ticks([0, 0.5, 1])
ax1.legend()

ax2.plot(
    e_qs.t_list,
    subsystem_entropy,
)
ax2.set_title("Entanglement Entropy")
ax2.set_ylabel(r"$e^{ \mathcal{S} ( \rho_A ) } \, / \, \frac{N}{2}$")

plt.tight_layout()
for ax in axs:
    ax.grid()

    # Move legend to the right
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()


def plot_basis_state_populations_2d(e_qs: EvolvingQubitSystem, log=False, log_limit=1e-5):
    quartile_index = int(len(e_qs.t_list) / 4)
    indices = [0, quartile_index, quartile_index * 2, quartile_index * 3, -1]

    # states = get_states(e_qs.N)
    states = get_states(e_qs.N, sparse=True)

    state_product_basis_indices_dict = defaultdict(list)
    number_of_bytes = math.ceil(e_qs.N / 8)
    for i in range(2 ** e_qs.N):
        y = bitarray()
        y.frombytes((i).to_bytes(number_of_bytes, byteorder='big'))

        state_product_basis_indices_dict[y.count()].append(i)
    state_product_basis_indices = np.array([
        _
        for i in range(e_qs.N, -1, -1)
        for _ in state_product_basis_indices_dict[i]
    ])


    labels = [get_label_from_state(state) for state in states]
    x = np.arange(len(labels))

    fig, axs = plt.subplots(len(indices), 1, sharex='all', figsize=(14, 8))

    for _i, i in enumerate(indices):
        ax = axs[_i]
        if not log:
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(log_limit, 1)
            ax.set_yscale('log', basey=10)
        ax.grid(axis='y')
        ax.set_ylabel(f"{e_qs.solved_t_list[i]:.2e}s")

        _solve_result_state = e_qs.solved_states[i]
        solve_result_state_populations = np.abs(_solve_result_state.flatten()) ** 2
        basis_state_populations = solve_result_state_populations[state_product_basis_indices]
        basis_state_populations += np.ones_like(basis_state_populations) * log_limit

        above_limit = np.count_nonzero(solve_result_state_populations > log_limit)
        print(f"above limit {log_limit:.0e} \t count: {above_limit:4d} \t ({above_limit / 2 ** e_qs.N:5.1%})")

        ax.fill_between(x, np.zeros_like(basis_state_populations), basis_state_populations, step='mid')

    if len(x) > 20:
        label_indices = [0, -1]
        plt.xticks(
            [x[i] for i in label_indices],
            [labels[i] for i in label_indices],
        )
    else:
        plt.xticks(x, labels)
    plt.tight_layout()
    plt.show()


plot_basis_state_populations_2d(e_qs, log=True)
