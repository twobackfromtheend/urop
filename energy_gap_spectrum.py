from collections import defaultdict

from scipy import constants
import numpy as np
from qutip import *
from hamiltonian import get_hamiltonian
from states import get_states, is_excited
import tqdm

PLOT_LIMITS = {
    2: [-15e6, 20e7],
    4: [-15e6, 20e7],
    8: [-15e6, 20e7],
}

PLOT_LIMITS = defaultdict(lambda: [-15e6, 15e6], PLOT_LIMITS)

# L = 2
# Omega = 3e6  # 3 megahertz
# Omega = 0
# Delta = [2 * constants.pi * 5e6 for _ in range(L)]
# V = 2 * constants.pi * 24e6


# Second paper
L = 8
a = 532e-9
# C6 = 2 * constants.pi * 0.4125e6 * (L * a) ** 6
C6 = 1.625e-60 / constants.hbar
V_L = C6 / ((L - 1) * a) ** 6

V = C6 / a ** 6
_plot_limits = [-3e4 * V_L, 23e4 * V_L]

# _plot_limits = PLOT_LIMITS[L] if V > 0 else [-PLOT_LIMITS[L][1], -PLOT_LIMITS[L][0]]
PLOT_POINTS = 10
detuning = np.linspace(_plot_limits[0], _plot_limits[1], PLOT_POINTS)
states = get_states(L)
print(f"Received {len(states)} states.")

# Energies for Omega = 0
all_energies = []
for i, Delta in tqdm.tqdm(enumerate(detuning)):
    H = get_hamiltonian(L, 0, Delta, V)
    energies = []
    for state in states:
        energy = expect(H, tensor(*state))
        energies.append(energy)
    all_energies.append(np.array(energies) / V_L)
    # eigenvalues, eigenstates = H.eigenstates()
    # all_energies.append(eigenvalues)
all_energies = np.array(all_energies)


# Energies for Omega != 0
# Omega = 3e6
# non_zero_omega_energies = []
# for i, Delta in enumerate(detuning):
#     H = get_hamiltonian(L, Omega, Delta, V)
#     eigenvalues, eigenstates = H.eigenstates()
#     non_zero_omega_energies.append(eigenvalues)
# non_zero_omega_energies = np.array(non_zero_omega_energies)


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))

# plt.plot(detuning, all_energies, 'b', alpha=0.6)
for i in reversed(range(len(states))):
    label = "".join(["e" if is_excited(spin) else "g" for spin in states[i]])
    # color = f'C{i}'
    color = 'g' if 'e' not in label else 'r' if 'g' not in label else 'grey'
    # plt.plot(detuning, all_energies[:, i], color=color, label=label, alpha=0.6)
    plt.plot(detuning / V_L / 1e4, all_energies[:, i] / 1e4, color=color, label=label, alpha=0.6)
    # plt.plot(detuning, non_zero_omega_energies[:, i], color=f'C{i}', ls=':', alpha=0.6)

    # detuning_index = int(PLOT_POINTS / len(states)) * i + int(PLOT_POINTS / 2 / len(states))
    # text_x = detuning[detuning_index]
    # text_y = all_energies[detuning_index, i]
    # plt.text(text_x, text_y, label, ha='center', color=f'C{i}', fontsize=16, fontweight='bold')

# plt.legend()

plt.grid()

plt.title(f"Plot of eigenvalues of $H$ with $V = {V:0.3e}$ ")
plt.xlabel("Detuning $\Delta$ (Hz)")
plt.ylabel("Energy (Hz)")
plt.tight_layout()

# plt.savefig(f"L_{L}_Omega_{Omega}.png")

# plt.savefig(f"L_{L}_V_{V:0.3e}_Omega_{Omega}.png", dpi=300)
plt.savefig(f"paper_2_L_{L}_V_{V:0.3e}.png", dpi=300)
plt.show()

