import numpy as np

import interaction_constants
from optimised_protocols.saver import save_protocol
from protocol_generator.interpolation_pg import InterpolationPG
from qubit_system.geometry import RegularLattice
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem
from qubit_system.utils.ghz_states import CustomGHZState

protocols = {
    (1, "std"): [1.21344545e+09, 1.23152656e+09, 1.10389513e+08, 1.02223011e+09, 1.86227838e+09, 1.20940030e+09],
    (1, "alt"): [441436.76288613, 17958550.53149551, 11187935.55716398, 24163677.64513641, 15150961.32302503,
                 10424052.76692684],
    (2, "std"): [2.00130092e+09, 1.57667386e+09, 1.36307468e+09, 2.27962022e+09, 2.16387636e+09, 2.19094970e+09],
    (2, "alt"): [3.51832094e+06, 1.65642207e+08, 8.60534422e+07, 2.14008038e+08, 1.61479519e+08, 1.02126899e+08],
    (3, "std"): [1.82728899e+09, 1.70821665e+09, 2.13164349e+09, 2.33794257e+09, 3.77499293e+09, 2.89800233e+09],
    (3, "alt"): [2.70546737e+08, 1.33471648e+08, 7.25294404e+07, 3.29288148e+08, 2.67062494e+08, 2.03814321e+08],
}
N = 12
t = 1e-6
LATTICE_SPACING = 1.5e-6

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

t_list = np.linspace(0, t, 3001)
for (D, ghz), protocol in protocols.items():
    if D == 1:
        shape = (12,)
        if ghz == "std":
            ghz_single_component = [True, True, True, True, True, True, True, True, True, True, True, True]
        elif ghz == "alt":
            ghz_single_component = [True, False, True, False, True, False, True, False, True, False, True, False]
    elif D == 2:
        shape = (4, 3)
        if ghz == "std":
            ghz_single_component = [True, True, True, True, True, True, True, True, True, True, True, True]
        elif ghz == "alt":
            ghz_single_component = [True, False, True, False, True, False, True, False, True, False, True, False]

    elif D == 3:
        shape = (3, 2, 2)
        if ghz == "std":
            ghz_single_component = [True, True, True, True, True, True, True, True, True, True, True, True]
        elif ghz == "alt":
            ghz_single_component = [True, False, False, True, False, True, True, False, True, False, False, True]
    else:
        raise ValueError

    geometry = RegularLattice(shape=shape, spacing=LATTICE_SPACING)
    ghz_state = CustomGHZState(N, ghz_single_component)

    pg = InterpolationPG(t_list, kind="cubic")
    Omega, Delta = pg.get_protocol(np.array(protocol))

    e_qs = EvolvingQubitSystem(
        N=N, V=C6, geometry=geometry,
        Omega=Omega,
        Delta=Delta,
        t_list=t_list,
        ghz_state=ghz_state
    )

    e_qs.solve()
    e_qs_fidelity = e_qs.get_fidelity_with('ghz')
    print(f"Solved system fidelity: {e_qs_fidelity:.5f}")
    save_protocol(
        f"12_BO_COMPARE_BO_{D}D_{ghz}_{e_qs_fidelity:.5f}",
        N=N, V=C6, geometry=geometry, GHZ_single_component=ghz_single_component,
        Omega=Omega,
        Delta=Delta,
        t_list=t_list,
        fidelity=e_qs.get_fidelity_with('ghz')
    )
