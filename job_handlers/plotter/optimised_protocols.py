import interaction_constants
from protocol_generator.interpolation_pg import InterpolationPG
from qubit_system.geometry import RegularLattice
from qubit_system.utils import states
from qubit_system.utils.ghz_states import CustomGHZState
import numpy as np
import quimb as q

OPTIMISED_PROTOCOLS = {
    1: {
        'std': [2.65629869e+07, 1.10137775e+09, 2.10554803e+07, 1.53833774e+09, 1.43818374e+09, 1.15576644e+09],
        # 0.96210
        # 'std': [7.22461680e+08, 6.49773034e+08, 6.00660122e+08, 1.13152837e+09, 1.52233991e+09, 1.30382513e+09],
        # 0.90795
        # 'std': [9.28880828e+08, 1.03690960e+09, 8.49647496e+08, 1.81531908e+09, 9.70763685e+08, 1.22157560e+09],
        # 0.89575
        # 'alt': [1034883.35720177, 10746002.32511696, 13138604.21549956, 12611089.34283306, 14807475.81352524,
        #         12823830.46326383],
        # 0.79958
        'alt': [2565801.10085787, 12372232.59839516, 6488618.09446081, 19508115.86734799, 13904167.59376139,
                10511527.37629483]
        # 0.87312
    },
    2: {
        # 'std': [1.59175109e+09, 4.40798493e+08, 8.22430687e+08, 1.52515077e+09, 2.72788764e+09, 2.08805395e+09],
        # 0.60493
        'std': [1.65811738e+09, 8.29998385e+08, 1.82769297e+09, 2.57907175e+09, 2.71117607e+09, 1.94975775e+09],
        # 0.97098
        'alt': [8.94101353e+07, 1.34436283e+08, 3.17347152e+07, 1.90844269e+08, 9.70544131e+07, 8.64859020e+07]
        # 0.99998
    }
}


def get_optimised_protocol(D: int, GHZ: str):
    N_RYD = 50
    C6 = interaction_constants.get_C6(N_RYD)

    LATTICE_SPACING = 1.5e-6

    N = 8
    t = 2e-6
    interpolation_timesteps = 3000
    t_list = np.linspace(0, t, interpolation_timesteps + 1)

    psi_0 = q.kron(*states.get_ground_states(N))

    assert D == 1 or D == 2, f"D has to be 1 or 2, not {D}"
    assert GHZ == "std" or GHZ == "alt", f"GHZ has to be std or alt, not {GHZ}"
    ghz_component = [True, True, True, True, True, True, True, True] if GHZ == "std" else None
    if D == 1:
        geometry_shape = (8,)
        if GHZ == "alt":
            ghz_component = [True, False, True, False, True, False, True, False]
    elif D == 2:
        geometry_shape = (4, 2)
        if GHZ == "alt":
            ghz_component = [True, False, False, True, True, False, False, True]
    else:
        raise ValueError(f"Unknown dimension D: {D}")

    geometry = RegularLattice(shape=geometry_shape, spacing=LATTICE_SPACING)
    ghz_state = CustomGHZState(N, ghz_component)
    protocol = OPTIMISED_PROTOCOLS[D][GHZ]
    protocol_generator = InterpolationPG(t_list, kind="quadratic")
    Omega, Delta = protocol_generator.get_protocol(np.array(protocol))

    return N, C6, geometry, t_list, psi_0, ghz_state, Omega, Delta
