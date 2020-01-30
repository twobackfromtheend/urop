import time

import matplotlib.pyplot as plt
import numpy as np
import quimb as q
import quimb.tensor as qtn
from quimb.tensor import MatrixProductState
from tqdm import tqdm

import interaction_constants
from protocol_generator.interpolation_pg import InterpolationPG
from qubit_system.geometry import RegularLattice
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem
from qubit_system.utils.ghz_states import CustomGHZState

optimised_protocols = {
    1: {
        'std': [2.65629869e+07, 1.10137775e+09, 2.10554803e+07, 1.53833774e+09, 1.43818374e+09, 1.15576644e+09],
        'alt': [1034883.35720177, 10746002.32511696, 13138604.21549956, 12611089.34283306, 14807475.81352524,
                12823830.46326383]

    },
    2: {
        'std': [1.59175109e+09, 4.40798493e+08, 8.22430687e+08, 1.52515077e+09, 2.72788764e+09, 2.08805395e+09],
        'alt': [8.94101353e+07, 1.34436283e+08, 3.17347152e+07, 1.90844269e+08, 9.70544131e+07, 8.64859020e+07]
    }
}

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

LATTICE_SPACING = 1.5e-6

N = 8
# geometry = RegularLattice(shape=(4, 2), spacing=LATTICE_SPACING)
# ghz_state = CustomGHZState(N, [True, False, False, True, True, False, False, True])
# protocol = optimised_protocols[2]['alt']
geometry = RegularLattice(shape=(8,), spacing=LATTICE_SPACING)
# ghz_state = CustomGHZState(N, [True, True, True, True, True, True, True, True])
# protocol = optimised_protocols[1]['std']
ghz_state = CustomGHZState(N, [True, False, True, False, True, False, True, False])
protocol = optimised_protocols[1]['alt']

t = 2e-6
interpolation_timesteps = 1000
t_list = np.linspace(0, t, interpolation_timesteps + 1)

protocol_generator = InterpolationPG(t_list, kind="quadratic")

Omega, Delta = protocol_generator.get_protocol(np.array(protocol))
e_qs = EvolvingQubitSystem(
    N, C6, geometry,
    Omega, Delta,
    t_list,
    ghz_state=ghz_state
)


start_time = time.time()
e_qs.solve()
print(f"Solved in {time.time() - start_time:.3f}s")
e_qs.plot(show=True)
ghz_fidelity = e_qs.get_fidelity_with("ghz")
print(f"fidelity: {ghz_fidelity}")


def get_spin_ham(e_qs: EvolvingQubitSystem, Omega: float, Delta: float):
    sx = q.pauli("X")
    sz = q.pauli("Z")
    qnum = (sz + q.identity(2)) / 2

    builder = qtn.SpinHam()
    # builder += Omega / 2, sx
    # builder -= Delta, qnum
    for i in range(e_qs.N):
        builder[i] += Omega / 2, sx
        builder[i] += -Delta, qnum

        for j in range(i):
            if i - j != 1:
                continue
            V = e_qs.V / e_qs.geometry.get_distance(i, j) ** 6
            builder[j, i] += V, qnum, qnum
    return builder.build_nni(e_qs.N), builder


def solve(e_qs) -> [q.qarray]:
    dt = e_qs.t_list[1]

    solved_matrix_product_states = []
    solved_t_list = [0]

    latest_state = state = MatrixProductState.from_dense(e_qs.psi_0, [2] * e_qs.N)
    latest_time = 0
    for i in tqdm(range(len(e_qs.Omega))):
        Omega = e_qs.Omega[i]
        Delta = e_qs.Delta[i]
        nni_ham, builder = get_spin_ham(e_qs, Omega, Delta)
        # quimb_ham = e_qs.get_hamiltonian(Omega, Delta)

        tebd = qtn.TEBD(
            latest_state,
            nni_ham,
            progbar=False
        )
        solve_points = np.linspace(0, dt, e_qs.solve_points_per_timestep + 1)[1:]  # Take away t=0 as a solve point
        for state in tebd.at_times(solve_points, tol=1e-10):
            solved_matrix_product_states.append(state)
        solved_t_list += (solve_points + latest_time).tolist()
        latest_state = state
        latest_time = solved_t_list[-1]

    solved_states = [e_qs.psi_0] + [mps.to_dense() for mps in solved_matrix_product_states]

    return solved_states, solved_t_list


states, _t_list = solve(e_qs)

e_qs.solved_states = states
e_qs.solved_t_list = _t_list
e_qs.evo = 1

e_qs.plot(savefig_name="123.png", show=True)


pass
