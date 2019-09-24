import numpy as np

import interaction_constants
from demonstration_utils import *
from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.utils.ghz_states import AlternatingGHZState

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

LATTICE_SPACING = 1.5e-6

print(f"C6: {C6:.3e}")
characteristic_V = C6 / (LATTICE_SPACING ** 6)
print(f"Characteristic V: {characteristic_V:.3e} Hz")

N = 8
timesteps = 50
t = 2e-6
geometry = RegularLattice1D(LATTICE_SPACING)
t_list = np.linspace(0, t, timesteps + 1)
ghz_state = AlternatingGHZState(N)

norm_t = t * characteristic_V
optim_result = get_optimised_controls(
    N, n_ts=timesteps, norm_t=norm_t, norm_V=1, target_state_symmetric=True,
    ghz_state=ghz_state,
    optim_kwargs={
        'init_pulse_type': "RND",
        'max_wall_time': 300, 'max_iter': 1000,
        # 'amp_lbound': 0, 'amp_ubound': 3,
        'amp_lbound': 0, 'amp_ubound': 0.2,
    }
)

# optim_result = get_optimised_controls(
#     N, n_ts=timesteps, norm_t=norm_t, norm_V=1, target_state_symmetric=True,
#     optim_kwargs={
#         'guess_pulse_type': "RND",
#         'max_wall_time': 120, 'max_iter': 1000,
#         'amp_ubound': 3, 'num_coeffs': timesteps
#     },
#     alg="CRAB"
# )
report_stats(optim_result, N, target_state_symmetric=True, ghz_state=ghz_state)
plot_optimresult(optim_result, N, t, C6, characteristic_V, geometry=RegularLattice1D(LATTICE_SPACING),
                 ghz_state=ghz_state)
import matplotlib.pyplot as plt

plt.show()
