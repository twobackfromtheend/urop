import numpy as np

import interaction_constants
from optimised_protocols.saver import save_protocol
from plots_creation.utils import _get_ghz_single_component_from_dimension
from protocol_generator.interpolation_pg import InterpolationPG
from qubit_system.geometry import *
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem
from qubit_system.utils.ghz_states import CustomGHZState

N = 8
t = 1e-6
LATTICE_SPACING = 1.5e-6

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

t_list = np.linspace(0, t, 3001)


def regular_lattice_saver():
    protocols = {
        1: {
            'std': [1.12038484e+09, 9.81854660e+07, 2.04816356e+08, 1.49816326e+09, 9.94404178e+08, 7.40829850e+08],
            'alt': [12457414.42490548, 12903872.9200179, 14027201.55622148, 21690495.74855748, 22873032.7832127,
                    12353572.28648302],
        },
        2: {
            'std': [1.10378910e+09, 1.40050082e+09, 2.47089287e+08, 1.27405221e+09, 2.59317139e+09, 1.54382328e+09],
            'alt': [1.09194278e+08, 1.18602159e+08, 1.14333625e+08, 1.57044351e+08, 1.57533640e+08, 7.02352731e+07],
        },
        3: {
            'std': [4.36624522e+08, 1.61912345e+09, 1.59368977e+09, 3.32036290e+09, 1.93348972e+09, 2.15484157e+09],
            'alt': [1.15013118e+08, 2.42755046e+08, 2.27086005e+08, 3.29858689e+08, 2.56155987e+08, 1.35135091e+08],
        },
    }

    for shape in [(8,), (4, 2), (2, 2, 2)]:
        D = len(shape)
        for ghz in ["std", "alt"]:
            ghz_is_alt = ghz == "alt"

            geometry = RegularLattice(shape=shape, spacing=LATTICE_SPACING)
            ghz_single_component = _get_ghz_single_component_from_dimension(D, ghz_is_alt)

            ghz_state = CustomGHZState(N, ghz_single_component)

            pg = InterpolationPG(t_list, kind="cubic")
            protocol = np.array(protocols[D][ghz])
            Omega, Delta = pg.get_protocol(protocol)

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
                f"BO_COMPARE_BO_{D}D_{ghz}_8",
                N=N, V=C6, geometry=geometry, GHZ_single_component=ghz_single_component,
                Omega=Omega,
                Delta=Delta,
                t_list=t_list,
                fidelity=e_qs.get_fidelity_with('ghz')
            )

            e_qs.plot(show=True)


def custom_geometry_saver():
    # name = "expanded_cross_alt"
    # geometry = DoubleRing(8, spacing=LATTICE_SPACING)
    # input_ = [4.56438420e+07, 9.07601900e+07, 4.80494153e+07, 1.68577544e+08, 1.17499188e+08, 1.05267001e+08]
    # ghz_single_component = [True, False, True, False, False, True, False, True]

    name = "unfolded_tetrahedron_alt"
    geometry = Star(8, spacing=LATTICE_SPACING)
    ghz_single_component = [True, False, True, False, False, True, False, True]
    input_ = [6.47956660e+08, 1.73824013e+07, 1.69317629e+08, 8.83276049e+08, 4.29156710e+08, 5.96229488e+08]

    pg = InterpolationPG(t_list, kind="cubic")
    Omega, Delta = pg.get_protocol(np.array(input_))
    ghz_state = CustomGHZState(N, ghz_single_component)

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
        f"BO_COMPARE_BO_{name}_8",
        N=N, V=C6, geometry=geometry, GHZ_single_component=ghz_single_component,
        Omega=Omega,
        Delta=Delta,
        t_list=t_list,
        fidelity=e_qs.get_fidelity_with('ghz')
    )

    e_qs.plot(show=True)


if __name__ == '__main__':
    # regular_lattice_saver()
    custom_geometry_saver()
