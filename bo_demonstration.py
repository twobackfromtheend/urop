from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from GPyOpt.methods import BayesianOptimization
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey

from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.qubit_system_classes_quimb import EvolvingQubitSystem
from qubit_system.utils.ghz_states_quimb import StandardGHZState, AlternatingGHZState

plt.rcParams['savefig.dpi'] = 600


def get_solved_episode(input_: np.ndarray,
                       N: int, V: float, geometry: BaseGeometry, t_list: np.ndarray,
                       interpolation_timesteps: int = 3000) -> EvolvingQubitSystem:
    timesteps = len(t_list) - 1

    Omega_params = input_[:timesteps]
    Delta_params = input_[timesteps:]

    _t_list = np.linspace(0, t_list[-1], interpolation_timesteps + 1)

    Omega_func: Callable[[float], float] = interp1d(t_list, np.hstack((Omega_params, Omega_params[-1])),
                                                    kind="quadratic")
    Omega_shape_window = tukey(interpolation_timesteps + 1, alpha=0.2)
    Omega = np.array([Omega_func(_t) * Omega_shape_window[_i] for _i, _t in enumerate(_t_list[:-1])])

    Delta_func: Callable[[float], float] = interp1d(t_list, np.hstack((Delta_params, Delta_params[-1])),
                                                    kind="quadratic")
    Delta = np.array([Delta_func(_t) for _t in _t_list[:-1]])

    e_qs = EvolvingQubitSystem(
        N, V, geometry,
        Omega, Delta,
        _t_list,
        # ghz_state=StandardGHZState(N)
        ghz_state=AlternatingGHZState(N)
    )
    e_qs.solve()
    return e_qs


def get_f(N: int, V: float, geometry: BaseGeometry, t_list: np.ndarray):
    def f(inputs: np.ndarray) -> np.ndarray:
        """
        :param inputs: 2-dimensional array
        :return: 2-dimentional array, one-evaluation per row
        """
        def get_figure_of_merit(*args):
            e_qs = get_solved_episode(*args, N=N, V=V, geometry=geometry, t_list=t_list)
            # return 1 - e_qs.get_fidelity_with("ghz")
            # component_products = e_qs.get_fidelity_with("ground") * e_qs.get_fidelity_with("excited") * 4
            component_products = e_qs.get_fidelity_with("ghz_component_1") * e_qs.get_fidelity_with("ghz_component_2") * 4
            # ghz_above_half = max(e_qs.get_fidelity_with("ghz") - 0.5, 0) * 2
            # return 1 - component_products * ghz_above_half
            return 1 - component_products
        return np.apply_along_axis(get_figure_of_merit, 1, inputs).reshape((-1, 1))

    return f


def get_domain(Omega_limits: Tuple[float, float], Delta_limits: Tuple[float, float], timesteps: int):
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
        'max_iter': 100,
        # 'max_time': 300,
    }
    print(f"optimisation_kwargs: {optimisation_kwargs}")
    bo.run_optimization(**optimisation_kwargs)

    print(f"Optimised result: {bo.fx_opt}")
    print(f"Optimised controls: {bo.x_opt}")
    return bo


def plot_result(bo: BayesianOptimization,
                N: int, V: float, geometry: BaseGeometry, t_list: np.ndarray):
    bo.plot_convergence()
    optimised_controls = bo.x_opt
    e_qs = get_solved_episode(input_=optimised_controls, N=N, V=V, geometry=geometry, t_list=t_list,
                              interpolation_timesteps=3000)
    print(f"fidelity: {e_qs.get_fidelity_with('ghz')}")
    e_qs.plot(show=True)


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

    N = 8
    timesteps = 4
    t = 2e-6
    geometry = RegularLattice1D(LATTICE_SPACING)
    t_list = np.linspace(0, t, timesteps + 1)

    f = get_f(
        N, C6,
        geometry=geometry,
        t_list=t_list
    )

    domain = get_domain(
        # Omega_limits=(0, characteristic_V),
        # Delta_limits=(0.85 * characteristic_V, characteristic_V),
        # Delta_limits=(0, 3 * characteristic_V),
        # Delta_limits=(0.7 * characteristic_V, 1.2 * characteristic_V),
        Omega_limits=(0, 0.1 * characteristic_V),
        Delta_limits=(0, 0.2 * characteristic_V),

        timesteps=timesteps
    )
    bo = optimise(f, domain)
    plot_result(bo, N, C6, geometry, t_list)
