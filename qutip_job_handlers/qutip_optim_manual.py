import numpy as np
from scipy import interpolate

import interaction_constants
from optimised_protocols.saver import save_protocol
from protocol_generator.interpolation_pg import InterpolationPG
from qubit_system.geometry import *
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem
from qubit_system.qubit_systems.static_qubit_system import StaticQubitSystem
from qubit_system.utils.ghz_states import CustomGHZState

N = 8
t = 1e-6
LATTICE_SPACING = 1.5e-6

geometry = RegularLattice(shape=(8,), spacing=LATTICE_SPACING)
ghz_single_component = [True, True, True, True, True, True, True, True]

# geometry = RegularLattice(shape=(4, 2), spacing=LATTICE_SPACING)
# ghz_single_component = [True, False, False, True, True, False, False, True]
ghz_state = CustomGHZState(N, ghz_single_component)

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

# s_qs = StaticQubitSystem(
#     N, C6,
#     geometry=geometry,
#     Omega=0, Delta=np.linspace(-0.8e9, 2.8e9, 10)
# )
# s_qs.plot()

t_list = np.linspace(0, t, 3001)

# Omega_func = interpolate.interp1d(
#     [0, t / 4, t * 3 / 4, t],
#     [0, 396.5e6, 396.5e6, 0],
#     kind="linear"
# )
# Delta_func = interpolate.interp1d(
#     [0, t],
#     [1.3e9, 1.2e9],
#     kind="linear"
# )

# Omega = np.array([Omega_func(_t) for _t in t_list[:-1]])
# Delta = np.array([Delta_func(_t) for _t in t_list[:-1]])


# RL RESULTS
Omega = [0.0, 246307925.46272278, 0.0, 0.0, 0.0, 190629606.48536682, 0.0, 0.0]
Delta = [1150822636.8427277, 1818000000.0, 1818000000.0, 606000000.0, 1818000000.0, 1423801659.822464, 1818000000.0, 1818000000.0]
Omega_func = interpolate.interp1d(
    np.linspace(0, t, 8),
    Omega,
    kind="previous"
)
Delta_func = interpolate.interp1d(
    np.linspace(0, t, 8),
    Delta,
    kind="previous"
)
Omega = np.array([Omega_func(_t) for _t in t_list[:-1]])
Delta = np.array([Delta_func(_t) for _t in t_list[:-1]])



# BO RESULTS
# pg = InterpolationPG(t_list, kind="cubic")
# Omega, Delta = pg.get_protocol(
#     # [7.55182664e+08, 1.11581681e+09, 1.00610185e+09, 1.49998750e+09,
#     #  8.31756017e+08, 1.21722990e+09]
#     [7.57238797e+07, 5.94473811e+08, 5.77848351e+07, 1.70690892e+09,
#      1.32805606e+09, 1.10148665e+09]
# )
# Omega = np.clip(Omega, 0, np.inf)



# Solve system


e_qs = EvolvingQubitSystem(
    N=N, V=C6, geometry=geometry,
    Omega=Omega,
    Delta=Delta,
    t_list=t_list,
    ghz_state=ghz_state
)

e_qs.solve()
print(f"Solved system fidelity: {e_qs.get_fidelity_with('ghz'):.5f}")

save_protocol(
    "RL_1D_STD_8",
    N=N, V=C6, geometry=geometry, GHZ_single_component=ghz_single_component,
    Omega=Omega,
    Delta=Delta,
    t_list=t_list,
    fidelity=e_qs.get_fidelity_with('ghz')
)

e_qs.plot(show=True)
