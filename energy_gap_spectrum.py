from scipy import constants
import numpy as np
from qutip import *
from hamiltonian import get_hamiltonian
from states import get_states, is_excited

L = 8
# Omega = 3e6  # 3 megahertz
Omega = 0
# Delta = [2 * constants.pi * 5e6 for _ in range(L)]
V = 2 * constants.pi * 24e6


detuning = np.linspace(-15e6, 15e6, 100)
states = get_states(L)

evals_mat = np.zeros((len(detuning), 2 ** L))
all_energies = []
for i, Delta in enumerate(detuning):
    H = get_hamiltonian(L, Omega, Delta, V)

    energies = []
    for state in states:
        energy = expect(H, tensor(*state))
        energies.append(energy)
    all_energies.append(energies)

all_energies = np.array(all_energies)


import matplotlib.pyplot as plt

# plt.plot(detuning, all_energies, 'b', alpha=0.6)
for i in reversed(range(len(states))):
    label = "".join(["e" if is_excited(spin) else "g" for spin in states[i]])
    plt.plot(detuning, all_energies[:, i], label=label, alpha=0.6)

    # text_x = 2e6 * i - 14e6
    # text_y = all_energies[int(len(detuning) / 2), i]
    # plt.text(text_x, text_y, label)

plt.ylim((-1e6, 8e6))

# plt.legend()

plt.grid()
plt.xlabel("Detuning $\Delta$ (Hz)")
plt.ylabel("Energy (Hz)")
plt.tight_layout()

# plt.savefig(f"L_{L}_Omega_{Omega}.png")
plt.show()

