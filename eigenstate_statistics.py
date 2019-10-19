import time

import matplotlib.pyplot as plt
import numpy as np

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
geometry = RegularLattice(shape=(4, 2), spacing=LATTICE_SPACING)
ghz_state = CustomGHZState(N, [True, False, False, True, True, False, False, True])
protocol = optimised_protocols[2]['alt']

t = 2e-6
interpolation_timesteps = 3000
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
ghz_fidelity = e_qs.get_fidelity_with("ghz")

print(f"fidelity: {ghz_fidelity}")

fig, axs = plt.subplots(4, 1, sharex='all', figsize=(12, 9))


