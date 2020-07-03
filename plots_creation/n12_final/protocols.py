import numpy as np

import interaction_constants
from optimised_protocols.saver import save_protocol
from protocol_generator.interpolation_pg import InterpolationPG
from protocol_generator.wide_interpolation_pg import WideInterpolationPG
from qubit_system.geometry import RegularLattice
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem
from qubit_system.utils.ghz_states import CustomGHZState

protocols = {
    # (1, "std", 0.5): [1.08081279e+09, 3.98468827e+08, 4.95220757e+08, 9.39029281e+08, 1.60076484e+09, 1.39668882e+09],
    # (1, "alt", 0.8): [12864164.00251228, 13929999.5630103, 16182438.28004114, 18748875.50105673, 19836048.07283583,
    #                   11165417.52171274],

    # (1, "std", 0.2): [1.21344545e+09, 1.23152656e+09, 1.10389513e+08, 1.02223011e+09, 1.86227838e+09, 1.20940030e+09],
    # (1, "alt", 0.2): [441436.76288613, 17958550.53149551, 11187935.55716398, 24163677.64513641, 15150961.32302503,
    #              10424052.76692684],
    # (2, "std", 0.2): [2.00130092e+09, 1.57667386e+09, 1.36307468e+09, 2.27962022e+09, 2.16387636e+09, 2.19094970e+09],
    # (2, "alt", 0.2): [3.51832094e+06, 1.65642207e+08, 8.60534422e+07, 2.14008038e+08, 1.61479519e+08, 1.02126899e+08],
    # (3, "std", 0.2): [1.82728899e+09, 1.70821665e+09, 2.13164349e+09, 2.33794257e+09, 3.77499293e+09, 2.89800233e+09],
    # (3, "alt", 0.2): [2.70546737e+08, 1.33471648e+08, 7.25294404e+07, 3.29288148e+08, 2.67062494e+08, 2.03814321e+08],
    (1, "std", 0.8): [4.48029301e+08, 7.19530539e+08, 1.72819439e+08, 9.14124784e+08, 1.33494897e+09, 1.20572155e+09],
    (2, "std", 0.8): [2.03656556e+09, 1.70305148e+09, 1.58486578e+09, 1.44662268e+09, 2.67199684e+09, 2.26869872e+09],
    (3, "std", 0.8): [1.99357238e+09, 1.89712671e+09, 1.41245085e+08, 3.42986973e+09, 2.68313113e+09, 2.96504632e+09],
}

N = 12
t = 1e-6
t = 1e-7
LATTICE_SPACING = 1.5e-6

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

t_list = np.linspace(0, t, 3001)
for (D, ghz, tukey_alpha), protocol in protocols.items():
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

    pg = WideInterpolationPG(t_list, kind="cubic", tukey_alpha=tukey_alpha)
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
    # protocol_name = f"12_BO_COMPARE_BO_WIDER_{D}D_{ghz}_{e_qs_fidelity:.5f}" if tukey_alpha > 0.2 else f"12_BO_COMPARE_BO_{D}D_{ghz}_{e_qs_fidelity:.5f}"
    protocol_name = f"12_BO_SHORT_STD_{D}D_{ghz}_{e_qs_fidelity:.5f}"
    save_protocol(
        protocol_name,
        N=N, V=C6, geometry=geometry, GHZ_single_component=ghz_single_component,
        Omega=Omega,
        Delta=Delta,
        t_list=t_list,
        fidelity=e_qs.get_fidelity_with('ghz')
    )
