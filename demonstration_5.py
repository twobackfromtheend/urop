# Investigating basis states populations over time
import pickle

import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from demonstration_utils import *
import interaction_constants
from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.qubit_system_classes import EvolvingQubitSystem
from qubit_system.utils.ghz_states import StandardGHZState
from qubit_system.utils.interpolation import get_hamiltonian_coeff_linear_interpolation
from qubit_system.utils.states import get_states, get_label_from_state, get_product_basis_states_index

plt.rc('text', usetex=False)
# plt.rc('text', usetex=True)
# plt.rc('font', family="serif", serif="CMU Serif")
# plt.rc('text.latex', preamble=r'\usepackage{upgreek}')


N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

LATTICE_SPACING = 1.5e-6

print(f"C6: {C6:.3e}")
characteristic_V = C6 / (LATTICE_SPACING ** 6)
print(f"Characteristic V: {characteristic_V:.3e} Hz")

norm_V = C6 / (LATTICE_SPACING ** 6) / characteristic_V


def plot_basis_state_populations_2d(e_qs: EvolvingQubitSystem, log=False, log_limit=1e-6):
    if e_qs.solve_result is None:
        e_qs.solve()
    quartile_index = int(len(e_qs.t_list) / 4)
    indices = [0, quartile_index, quartile_index * 2, quartile_index * 3, -1]

    states = get_states(e_qs.N)
    labels = [get_label_from_state(state) for state in states]
    x = np.arange(len(labels))

    fig, axs = plt.subplots(len(indices) , 1, sharex='all', figsize=(10, 8))

    for _i, i in enumerate(indices):
        ax = axs[_i]
        if not log:
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(log_limit, 1)
            ax.set_yscale('log', basey=10)
        ax.grid(axis='y')
        ax.set_ylabel(f"{e_qs.solve_result.times[i]:.2e}s")

        basis_state_populations = []

        _solve_result_state = e_qs.solve_result.states[i]
        solve_result_state_populations = np.abs(_solve_result_state.data.toarray().flatten()) ** 2
        for state in states:
            state_product_basis_index = get_product_basis_states_index(state)
            basis_state_population = solve_result_state_populations[state_product_basis_index]
            basis_state_populations.append(basis_state_population)
            if log:
                basis_state_population += log_limit
        ax.bar(x=x, height=basis_state_populations)
    plt.xticks(x, labels)
    plt.tight_layout()
    plt.show()


def plot_basis_state_populations_3d(e_qs: EvolvingQubitSystem):
    if e_qs.solve_result is None:
        e_qs.solve()

    states = get_states(e_qs.N)

    times = []
    heights = []
    xs = []

    for i, _solve_result_state in enumerate(e_qs.solve_result.states):
        plt.ylim(0, 1)
        plt.grid(axis='y')

        solve_result_state_populations = np.abs(_solve_result_state.data.toarray().flatten()) ** 2
        for _x, state in enumerate(states):
            state_product_basis_index = get_product_basis_states_index(state)
            basis_state_population = solve_result_state_populations[state_product_basis_index]
            times.append(e_qs.solve_result.times[i])
            xs.append(_x)
            heights.append(basis_state_population)

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection="3d")
    dt = e_qs.solve_result.times[1] - e_qs.solve_result.times[0]
    ax.bar3d(xs, times, z=0, dx=0.8, dy=dt, dz=heights)
    labels = [get_label_from_state(state) for state in states]
    x = np.arange(len(labels))
    plt.xticks(x, labels)

    ax.yaxis.set_major_formatter(ticker.EngFormatter('s'))
    plt.ylabel('Time')
    plt.show()


# N = 4
# t = 0.05e-6
# e_qs = EvolvingQubitSystem(
#     N=N, V=C6, geometry=RegularLattice1D(spacing=LATTICE_SPACING),
#     Omega=get_hamiltonian_coeff_linear_interpolation([0, t / 5, t * 4 / 5, t], [0, 750e6, 750e6, 0]),
#     Delta=get_hamiltonian_coeff_linear_interpolation([0, t], [1.5e9, 1e9]),
#     t_list=np.linspace(0, t, 50),
#     ghz_state=StandardGHZState(N)
# )
# solve_and_print_stats(e_qs)
# plot_basis_state_populations_2d(e_qs, log=True)
# plot_basis_state_populations_3d(e_qs)

# with open("reinforcement_learning/results/20190821_135329.pkl", "rb") as f:
#     data = pickle.load(f)
#
# t_list = data['evolving_qubit_system_kwargs']['t_list']
# solve_t_list = np.linspace(t_list[0], t_list[-1], 100)
#
# data['evolving_qubit_system_kwargs'].pop('t_list')
# e_qs = EvolvingQubitSystem(
#     **data['evolving_qubit_system_kwargs'],
#     Omega=get_hamiltonian_coeff_linear_interpolation(
#         t_list,
#         data['protocol'].Omega,
#     ),
#     Delta=get_hamiltonian_coeff_linear_interpolation(
#         t_list,
#         data['protocol'].Delta,
#     ),
#     t_list=solve_t_list,
# )
# solve_and_print_stats(e_qs)
# plot_basis_state_populations_2d(e_qs, log=True)

# N = 6
# t = 2e-6
# e_qs = EvolvingQubitSystem(
#     N=N, V=C6, geometry=RegularLattice1D(spacing=LATTICE_SPACING),
#     Omega=get_hamiltonian_coeff_linear_interpolation([0, t / 2, t], [0, 751.23e6, 0]),
#     Delta=get_hamiltonian_coeff_linear_interpolation([0, t], [1.3e9, 1.14e9]),
#     t_list=np.linspace(0, t, 300),
#     ghz_state=StandardGHZState(N)
# )
# solve_and_print_stats(e_qs)
# plot_basis_state_populations_2d(e_qs)


N = 8
t = 0.5e-6
e_qs = EvolvingQubitSystem(
    N=N, V=C6, geometry=RegularLattice1D(spacing=LATTICE_SPACING),
    Omega=get_hamiltonian_coeff_linear_interpolation([0, t / 6, t * 5 / 6, t], [0, 480e6, 490e6, 0]),
    Delta=get_hamiltonian_coeff_linear_interpolation([0, t], [1.5e9, 1.2e9]),
    t_list=np.linspace(0, t, 100),
    ghz_state=StandardGHZState(N)
)
solve_and_print_stats(e_qs)
plot_basis_state_populations_2d(e_qs, log=True)


# N = 8, t = 2e-6
# N = 8
#
# with open("reinforcement_learning/results/20190814_215943.pkl", "rb") as f:
#     data = pickle.load(f)
#
# t_list = data['evolving_qubit_system_kwargs']['t_list']
# solve_t_list = np.linspace(t_list[0], t_list[-1], 100)
#
# data['evolving_qubit_system_kwargs'].pop('t_list')
# e_qs = EvolvingQubitSystem(
#     **data['evolving_qubit_system_kwargs'],
#     Omega=get_hamiltonian_coeff_linear_interpolation(
#         t_list,
#         data['protocol'].Omega,
#     ),
#     Delta=get_hamiltonian_coeff_linear_interpolation(
#         t_list,
#         data['protocol'].Delta,
#     ),
#     t_list=solve_t_list,
# )
# solve_and_print_stats(e_qs)
# plot_basis_state_populations_2d(e_qs, log=True)
