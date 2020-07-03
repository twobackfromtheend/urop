import numpy as np

import interaction_constants
from optimised_protocols.saver import save_protocol
from protocol_generator.interpolation_pg import InterpolationPG
from protocol_generator.wide_interpolation_pg import WideInterpolationPG
from qubit_system.geometry import RegularLattice
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem
from qubit_system.utils.ghz_states import CustomGHZState

protocols = [
    # {
    #     'N': 8,
    #     'D': 2,
    #     't': 1e-7,
    #     'ghz': "std",
    #     'tukey_alpha': 0.8,
    #     'shape': (4, 2),
    #     'ghz_single_component': [True, True, True, True, True, True, True, True],
    #     'protocol': [4.06166851e+07, 6.92155159e+08, 1.61291452e+09, 2.70461695e+09, 1.96056433e+09, 1.83774303e+09],
    #     'prefix': 'entanglement_entropy_ramp_',
    # },
    {
        'N': 8,
        'D': 2,
        't': 1e-7,
        'ghz': "std",
        'tukey_alpha': 0.8,
        'shape': (4, 2),
        'ghz_single_component': [True, True, True, True, True, True, True, True],
        'protocol': [1.02953245e+08, 7.29131015e+08, 1.39144882e+09, 1.57206525e+09, 2.74812429e+09, 2.08432039e+09],
        'prefix': 'entanglement_entropy_ramp_2',
    },
    # {
    #     'N': 8,
    #     'D': 2,
    #     't': 1e-7,
    #     'ghz': "std",
    #     'tukey_alpha': 0.8,
    #     'shape': (4, 2),
    #     'ghz_single_component': [True, True, True, True, True, True, True, True],
    #     'protocol': [3.86930566e+08, 9.03028074e+07, 1.14008420e+09, 1.00450300e+09, 2.04728051e+09, 1.86689589e+09],
    #     'prefix': 'entanglement_entropy_ramp_',
    # },
    # {
    #     'N': 12,
    #     'D': 2,
    #     't': 1e-7,
    #     'ghz': "std",
    #     'tukey_alpha': 0.8,
    #     'shape': (4, 3),
    #     'ghz_single_component': [True, True, True, True, True, True, True, True, True, True, True, True],
    #     'protocol': [9.50945970e+08, 5.85600603e+08, 1.73877135e+09, 2.46653328e+09, 2.10610827e+09, 2.20269712e+09],
    #     'prefix': 'entanglement_entropy_ramp_',
    # },
    # {
    #     'N': 16,
    #     'D': 2,
    #     't': 1e-7,
    #     'ghz': "std",
    #     'tukey_alpha': 0.8,
    #     'shape': (4, 4),
    #     'ghz_single_component': [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
    #     'protocol': [3.16370289e+08, 1.75109737e+09, 1.73687868e+09, 3.16599854e+09, 3.31700205e+09, 2.45619104e+09],
    #     'prefix': 'entanglement_entropy_ramp_',
    # },
    # {
    #     'N': 8,
    #     'D': 2,
    #     't': 1e-6,
    #     'ghz': "alt",
    #     'tukey_alpha': 0.2,
    #     'shape': (4, 2),
    #     'ghz_single_component': [True, False, False, True, True, False, False, True],
    #     'protocol': [6.95817661e+07, 1.32444920e+08, 7.54100681e+07, 1.98829394e+08, 1.44806717e+08, 1.01798848e+08],
    #     'prefix': 'entanglement_entropy_ramp_',
    # },
    # {
    #     'N': 12,
    #     'D': 2,
    #     't': 1e-6,
    #     'ghz': "alt",
    #     'tukey_alpha': 0.2,
    #     'shape': (4, 3),
    #     'ghz_single_component': [True, False, True, False, True, False, True, False, True, False, True, False],
    #     'protocol': [1.17507163e+07, 1.67961292e+08, 3.15499580e+07, 2.74878754e+08, 1.76018772e+08, 1.29161574e+08],
    #     'prefix': 'entanglement_entropy_ramp_',
    # },
    # {
    #     'N': 16,
    #     'D': 2,
    #     't': 1e-6,
    #     'ghz': "alt",
    #     'tukey_alpha': 0.2,
    #     'shape': (4, 4),
    #     'ghz_single_component': [True, False, True, False, False, True, False, True, True, False, True, False, False, True, False, True],
    #     'protocol': [1.52846931e+08, 1.80192385e+08, 4.70101731e+07, 2.79508632e+08, 1.51929746e+08, 1.62403870e+08],
    #     'prefix': 'entanglement_entropy_ramp_',
    # },
]

LATTICE_SPACING = 1.5e-6

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

for protocol_info in protocols:
    N = protocol_info['N']
    D = protocol_info['D']
    t = protocol_info['t']
    ghz = protocol_info['ghz']
    tukey_alpha = protocol_info['tukey_alpha']
    shape = protocol_info['shape']
    ghz_single_component = protocol_info['ghz_single_component']
    protocol = protocol_info['protocol']
    prefix = protocol_info['prefix']

    t_list = np.linspace(0, t, 3001)

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
    protocol_name = f"{prefix}_{N}_{D}D_{ghz}_{e_qs_fidelity:.5f}"
    save_protocol(
        protocol_name,
        N=N, V=C6, geometry=geometry, GHZ_single_component=ghz_single_component,
        Omega=Omega,
        Delta=Delta,
        t_list=t_list,
        fidelity=e_qs.get_fidelity_with('ghz')
    )
