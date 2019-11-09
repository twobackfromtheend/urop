import random
import time

import matplotlib.pyplot as plt
import numpy as np
import quimb as q
import scipy
from scipy.sparse.linalg import eigsh

from qubit_system.utils.ghz_states import CustomGHZState, StandardGHZState, AlternatingGHZState

N = 12
ket_0 = q.ket([0, 1])
ket_1 = q.ket([1, 0])
P = ket_0 @ ket_0.H
# X = ket_0 @ ket_1.H + ket_1 @ ket_0.H
X = q.pauli('x')

print("Generating PXP Hamiltonian")
dims = [2] * N
pxp_hamiltonian = 0
for i in range(N):
    # Open boundary conditions
    if i == 0:
        X_i1 = q.ikron(X, dims=dims, inds=i, sparse=True)
        P_i2 = q.ikron(P, dims=dims, inds=i + 1, sparse=True)
        pxp_hamiltonian += X_i1 @ P_i2
        continue
    elif i == N - 1:
        P_i0 = q.ikron(P, dims=dims, inds=i - 1, sparse=True)
        X_i1 = q.ikron(X, dims=dims, inds=i, sparse=True)
        pxp_hamiltonian += P_i0 @ X_i1
        continue
    P_i0 = q.ikron(P, dims=dims, inds=i - 1, sparse=True)
    X_i1 = q.ikron(X, dims=dims, inds=i, sparse=True)
    P_i2 = q.ikron(P, dims=dims, inds=i + 1, sparse=True)
    pxp_hamiltonian += P_i0 @ X_i1 @ P_i2

print("PXP Hamiltonian generated.")

start_time = time.time()
pxp_eigenenergies, pxp_eigenstates = q.eigh(q.qu(pxp_hamiltonian, sparse=False))
print(f"Retrieved eigenstates in {time.time() - start_time:.3f}s")

start_time = time.time()
pxp_eigenenergies, pxp_eigenstates = q.eigh(q.qu(pxp_hamiltonian, sparse=False), autoblock=True)
print(f"Retrieved eigenstates in {time.time() - start_time:.3f}s")

start_time = time.time()
w, v = eigsh(pxp_hamiltonian, k=2 ** N - 2, which='SA')
print(f"Retrieved eigenstates in {time.time() - start_time:.3f}s")
print(w)

pxp_eigenstates = pxp_eigenstates.T

asfasfas

ghz_state_alt = AlternatingGHZState(N)
ghz_state_std = StandardGHZState(N)
z2, z2_prime = ghz_state_alt._get_components()
ghz_state_z3 = CustomGHZState(N, [n % 3 == 0 for n in range(N)])
ghz_state_z4 = CustomGHZState(N, [n % 4 == 0 for n in range(N)])


# ghz_state_z3_10010010 = CustomGHZState(8, [True, False, False, True, False, False, True, False])
# ghz_state_z4_10001000 = CustomGHZState(8, [True, False, False, False, True, False, False, False])
# ghz_state_random_1 = CustomGHZState(8, [True, False, False, True, False, False, True, False])
# ghz_state_random_11100101 = CustomGHZState(8, [True, True, True, False, False, True, False, True])
# ghz_state_random_01000010 = CustomGHZState(8, [False, True, False, False, False, False, True, False])
# ghz_state_random_10111100 = CustomGHZState(8, [True, False, True, True, True, True, False, False])
# ghz_state_random_11010010 = CustomGHZState(8, [True, True, False, True, False, False, True, False])


def get_random_state(N: int):
    random_bits = random.getrandbits(N)
    bit_string = f"{random_bits:b}".zfill(N)
    ghz_state = CustomGHZState(N, [letter == "1" for letter in bit_string])
    return f"ghz_state_random_{bit_string}", ghz_state.get_state_tensor()


random_states = [get_random_state(N) for _ in range(5)]
# for name, state in [
#     ('z2', z2),
#     ('z2_prime', z2_prime),
#     ('ghz_alt', ghz_state_alt.get_state_tensor()),
#     ('ghz_std', ghz_state_std.get_state_tensor()),
#     ('ghz_state_z3_10010010', ghz_state_z3_10010010.get_state_tensor()),
#     ('ghz_state_z4_10001000', ghz_state_z4_10001000.get_state_tensor()),
#     ('ghz_state_random_11100101', ghz_state_random_11100101.get_state_tensor()),
#     ('ghz_state_random_01000010', ghz_state_random_01000010.get_state_tensor()),
#     ('ghz_state_random_10111100', ghz_state_random_10111100.get_state_tensor()),
#     ('ghz_state_random_11010010', ghz_state_random_11010010.get_state_tensor()),
# ]:

for name, state in [
                       ('z2', z2),
                       ('ghz_state_alt', ghz_state_alt.get_state_tensor()),
                       ('ghz_state_std', ghz_state_std.get_state_tensor()),
                       ('ghz_state_z3', ghz_state_z3.get_state_tensor()),
                       ('ghz_state_z4', ghz_state_z4.get_state_tensor()),
                   ] + random_states:
    state = q.qu(state, sparse=False)
    eigenstate_overlaps = []
    for eigenenergy, eigenvector in zip(pxp_eigenenergies, pxp_eigenstates):
        eigenstate_overlap = q.fidelity(state, eigenvector)
        eigenstate_overlaps.append(eigenstate_overlap)
    eigenstate_overlaps = np.array(eigenstate_overlaps)
    fig, axs = plt.subplots(
        2, 1,
        gridspec_kw={
            'hspace': 0.01, 'height_ratios': [10, 1],
            'top': 0.9, 'bottom': 0.1, 'left': 0.12, 'right': 0.95,
        }
    )
    ax = axs[0]
    ax.set_yscale('log', basey=10)
    y_lim = 1e-8
    ax.set_ylim((y_lim, 2))
    ax.scatter(pxp_eigenenergies, eigenstate_overlaps, alpha=0.3, s=12)

    ax1 = axs[1]
    masked_i = eigenstate_overlaps < y_lim
    ax1.scatter(pxp_eigenenergies[masked_i], eigenstate_overlaps[masked_i], alpha=0.3, s=12)
    ax1.set_ylim((0, y_lim))
    ax1.set_yticks([0])

    ax1.set_xlabel("$E$")
    ax.set_ylabel(r"${\left| \langle \psi_{\mathrm{title}} | \psi_k \rangle \right| }^2$")

    fig.suptitle(name)
    plt.savefig(f"N{N}_{name}.png", dpi=300)
    # plt.show()

print(len(pxp_eigenstates))

# pxp_hamiltonian =
