from functools import partial
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import quimb as q
import tqdm
from GPyOpt.methods import BayesianOptimization
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey

from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.geometry.double_ring import DoubleRing
from qubit_system.geometry.regular_lattice_2d import RegularLattice2D
from qubit_system.geometry.star import Star
from qubit_system.qubit_system_classes_quimb import EvolvingQubitSystem, StaticQubitSystem
from qubit_system.utils.ghz_states_quimb import BaseGHZState, CustomGHZState

plt.rcParams['savefig.dpi'] = 600


def get_solved_episode(input_: np.ndarray,
                       N: int, V: float, geometry: BaseGeometry, t_list: np.ndarray,
                       ghz_state: BaseGHZState,
                       interpolation_timesteps: int = 3000) -> EvolvingQubitSystem:
    timesteps = len(t_list) - 1

    Omega_params = input_[:timesteps]
    Delta_params = input_[timesteps:]

    _t_list = np.linspace(0, t_list[-1], interpolation_timesteps + 1)
    interp = partial(interp1d,
                     # kind="cubic",
                     kind="quadratic",
                     # kind="linear",
                     # kind="previous",
                     )

    Omega_func: Callable[[float], float] = interp(t_list, np.hstack((Omega_params, Omega_params[-1])))
    Omega_shape_window = tukey(interpolation_timesteps + 1, alpha=0.2)
    Omega = np.array([Omega_func(_t) * Omega_shape_window[_i] for _i, _t in enumerate(_t_list[:-1])])

    Delta_func: Callable[[float], float] = interp(t_list, np.hstack((Delta_params, Delta_params[-1])))
    Delta = np.array([Delta_func(_t) for _t in _t_list[:-1]])

    e_qs = EvolvingQubitSystem(
        N, V, geometry,
        Omega, Delta,
        _t_list,
        ghz_state=ghz_state
    )
    e_qs.solve()
    return e_qs


def get_f(N: int, V: float, geometry: BaseGeometry, t_list: np.ndarray, ghz_state: BaseGHZState):
    def f(inputs: np.ndarray) -> np.ndarray:
        """
        :param inputs: 2-dimensional array
        :return: 2-dimentional array, one-evaluation per row
        """

        def get_figure_of_merit(*args):
            e_qs = get_solved_episode(*args, N=N, V=V, geometry=geometry, t_list=t_list, ghz_state=ghz_state)
            # return 1 - e_qs.get_fidelity_with("ghz")
            # component_products = e_qs.get_fidelity_with("ground") * e_qs.get_fidelity_with("excited") * 4
            component_products = e_qs.get_fidelity_with("ghz_component_1") * e_qs.get_fidelity_with(
                "ghz_component_2") * 4
            # return 1 - component_products
            ghz_above_half = max(e_qs.get_fidelity_with("ghz") - 0.5, 0) * 2
            return 1 - component_products * ghz_above_half

        return np.apply_along_axis(get_figure_of_merit, 1, inputs).reshape((-1, 1))

    return f


def get_domain(Omega_limits: Tuple[float, float], Delta_limits: Tuple[float, float], timesteps: int) -> List[dict]:
    return [
               {
                   'name': f'var_{i}',
                   'type': 'continuous',
                   'domain': Omega_limits
               }
               for i in range(timesteps)
           ] + [
               {
                   'name': f'var_{i}',
                   'type': 'continuous',
                   'domain': Delta_limits
               }
               for i in range(timesteps)
           ]


def optimise(f: Callable, domain: List[dict]):
    """
    :param f:
        function to optimize.
        It should take 2-dimensional numpy arrays as input and return 2-dimensional outputs (one evaluation per row).
    :param domain:
        list of dictionaries containing the description of the inputs variables
        (See GPyOpt.core.task.space.Design_space class for details).
    :return:
    """

    bo_kwargs = {
        'domain': domain,  # box-constraints of the problem
        'acquisition_type': 'EI',  # Selects the Expected improvement
        # 'initial_design_numdata': 4,  # Number of initial points
        'initial_design_numdata': 4 * len(domain),  # Number of initial points
        'exact_feval': True
    }
    print(f"bo_kwargs: {bo_kwargs}")

    bo = BayesianOptimization(
        f=f,
        # maximise=True,
        # initial_design_type='latin',
        # model_type="sparseGP",
        # batch_size=6,
        **bo_kwargs
    )

    optimisation_kwargs = {
        'max_iter': 50,
        # 'max_time': 300,
    }
    print(f"optimisation_kwargs: {optimisation_kwargs}")
    bo.run_optimization(**optimisation_kwargs)

    print(f"Optimised result: {bo.fx_opt}")
    print(f"Optimised controls: {bo.x_opt}")
    return bo


def plot_result(bo: BayesianOptimization,
                N: int, V: float, geometry: BaseGeometry, t_list: np.ndarray, ghz_state: BaseGHZState,
                **kwargs):
    # bo.plot_convergence()
    optimised_controls = bo.x_opt
    e_qs = get_solved_episode(input_=optimised_controls, N=N, V=V, geometry=geometry, t_list=t_list,
                              ghz_state=ghz_state,
                              interpolation_timesteps=3000)
    print(f"fidelity: {e_qs.get_fidelity_with('ghz')}")
    plot_kwargs = {'show': True}
    e_qs.plot(**{**plot_kwargs, **kwargs})


def get_crossing(ghz_state: BaseGHZState, geometry: BaseGeometry, N: int, V: float):
    ghz_1 = ghz_state._get_components()[1]
    Delta_range = np.linspace(-characteristic_V * 5, characteristic_V * 5, 50)

    s_qs = StaticQubitSystem(
        N=N, V=V, geometry=geometry,
        Omega=0, Delta=Delta_range,
    )

    energies = []
    for detuning in Delta_range:
        H = s_qs.get_hamiltonian(detuning)
        energy = q.expec(H, ghz_1).real
        energies.append(energy)
    energies = np.array(energies)

    def find_root(x: np.ndarray, y: np.ndarray):
        """
        Finds crossing (where y equals 0), given that x, y is roughly linear.
        """
        if (y == 0).all():
            return np.nan
        _right_bound = (y < 0).argmax()
        _left_bound = _right_bound - 1
        crossing = y[_left_bound] / (y[_left_bound] - y[_right_bound]) \
                   * (x[_right_bound] - x[_left_bound]) + x[_left_bound]
        return crossing

    crossing = find_root(Delta_range, energies)
    print(f"Found crossing: {crossing:.3e}")

    return crossing


if __name__ == '__main__':
    from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
    import interaction_constants

    N_RYD = 50
    C6 = interaction_constants.get_C6(N_RYD)

    LATTICE_SPACING = 1.5e-6

    print(f"C6: {C6:.3e}")
    characteristic_V = C6 / (LATTICE_SPACING ** 6)
    print(f"Characteristic V: {characteristic_V:.3e} Hz")

    # N = 8
    # geometry = RegularLattice1D(LATTICE_SPACING)
    #
    # s_qs = StaticQubitSystem(
    #     N=N, V=C6,
    #     # geometry=RegularLattice1D(spacing=LATTICE_SPACING),
    #     geometry=geometry,
    #     Omega=0, Delta=np.linspace(-1e9, 3e9, 3)
    # )
    # s_qs.plot_detuning_energy_levels(
    #     plot_state_names=False, show=True,
    #     highlight_states_by_label=[
    #         # ''.join(['eg' for _ in range(int(N / 2))]),
    #         # 'egeggege',
    #         ''.join(['e' for _ in range(N)])
    #     ]
    # )

    N = 2
    timesteps = 3
    t = 2e-6
    t_list = np.linspace(0, t, timesteps + 1)

    configurations: List[Tuple[BaseGeometry, BaseGHZState, str]] = [
        #
        # (RegularLattice1D(LATTICE_SPACING),
        #  CustomGHZState(N, [True, True, True, True, True, True, True, True]),
        #  "1d_std"),

        # (RegularLattice1D(LATTICE_SPACING),
        #  CustomGHZState(N, [True, False, True, False, True, False, True, False]),
        #  ((0, 0.1 * characteristic_V), (0, 0.2 * characteristic_V)),
        #  "1d_alt"),

        # (RegularLattice2D((4, 2), spacing=LATTICE_SPACING),
        #  CustomGHZState(N, [True, True, True, True, True, True, True, True]),
        #  ((0, 0.1 * characteristic_V), (0, 0.2 * characteristic_V)),
        #  "2d_std"),
        #
        # (RegularLattice2D((4, 2), spacing=LATTICE_SPACING),
        #  CustomGHZState(N, [True, False, False, True, True, False, False, True]),
        #  "2d_alt"),
        #
        # (RegularLattice3D((2, 2, 2), spacing=LATTICE_SPACING),
        #  CustomGHZState(N, [True, True, True, True, True, True, True, True]),
        #  "3d_std"),
        #
        # (RegularLattice3D((2, 2, 2), spacing=LATTICE_SPACING),
        #  CustomGHZState(N, [True, False, False, True, False, True, True, False]),
        #  "3d_alt"),
        #
        # (DoubleRing(8, spacing=LATTICE_SPACING),
        #  CustomGHZState(N, [True, True, True, True, True, True, True, True]),
        #  "2D_ring_std"),
        #
        # (DoubleRing(8, spacing=LATTICE_SPACING),
        #  CustomGHZState(N, [True, False, True, False, False, True, False, True]),
        #  "2D_ring_alt"),
        #
        # (Star(8, spacing=LATTICE_SPACING),
        #  CustomGHZState(N, [True, True, True, True, True, True, True, True]),
        #  "2D_star_std"),
        #
        # (Star(8, spacing=LATTICE_SPACING),
        #  CustomGHZState(N, [True, False, True, False, False, True, False, True]),
        #  "2D_star_alt"),

        # (RegularLattice2D((4, 3), spacing=LATTICE_SPACING),
        #  CustomGHZState(N, [True, False, True, False, True, False, True, False, True, False, True, False]),
        #  "2d_alt_n12"),
        # (RegularLattice2D((4, 4), spacing=LATTICE_SPACING),
        #  CustomGHZState(N, [True, False, True, False, False, True, False, True, True, False, True, False, False, True, False, True]),
        #  "2d_alt_n16"),

        # (RegularLattice1D(LATTICE_SPACING),
        #  CustomGHZState(N, [True, True, True, True]),
        #  "1d_std_n4"),

        (RegularLattice1D(LATTICE_SPACING),
         CustomGHZState(N, [True, True]),
         "1d_std_n2"),
    ]

    REPEATS = 2
    for i, (geometry, ghz_state, name) in enumerate(tqdm.tqdm(configurations)):
        crossing = get_crossing(ghz_state, geometry, N, C6)
        Omega_limits = (0, crossing)
        Delta_limits = (0.5 * crossing, 1.5 * crossing)
        domain = get_domain(Omega_limits, Delta_limits, timesteps)

        fidelities = []
        for repeat in range(REPEATS):
            f = get_f(
                N, C6,
                geometry=geometry,
                t_list=t_list,
                ghz_state=ghz_state
            )
            bo = optimise(f, domain)
            optimised_controls = bo.x_opt
            e_qs = get_solved_episode(input_=optimised_controls, N=N, V=C6, geometry=geometry, t_list=t_list,
                                      ghz_state=ghz_state,
                                      interpolation_timesteps=3000)
            fidelity = e_qs.get_fidelity_with('ghz')
            # print(f"fidelity: {fidelity}")
            plot_result(bo, N, C6, geometry, t_list, ghz_state, show=True)
            # plot_result(bo, N, C6, geometry, t_list, ghz_state, savefig_name=f"bo_demo_{name}_{repeat}", show=False)
            fidelities.append(fidelity)

        fidelities = np.array(fidelities)
        print(f"\n\n{i}"
              f"\n\t {fidelities}"
              f"\n\t {fidelities.mean()}"
              f"\n\t {fidelities.std()}")
