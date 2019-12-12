from typing import Tuple, List

import qutip
import qutip.control.pulseoptim as cpo
from qutip.control.optimresult import OptimResult
from scipy.interpolate import interp1d

from optimised_protocols.saver import save_protocol
from qubit_system.geometry import RegularLattice
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem
from qubit_system.utils.ghz_states import CustomGHZState as QuimbGHZState
from qutip_job_handlers.hamiltonian import QutipSpinHamiltonian
from qutip_job_handlers.qutip_states import BaseGHZState, CustomGHZState


def get_hamiltonian_variables(N: int, V: float, geometry: RegularLattice):
    e_qs = QutipSpinHamiltonian(N=N)
    # noinspection PyTypeChecker
    hamiltonian = e_qs.get_hamiltonian(V, geometry, Omega=None, Delta=None)

    H_d = hamiltonian[0]  # "drift": time-independent part
    H_c = [hamiltonian[1][0], hamiltonian[2][0]]  # "control": time-dependent parts
    return H_d, H_c


def get_optimised_controls(N: int, timesteps: int,
                           norm_t: float, characteristic_V: float,
                           geometry: RegularLattice,
                           ghz_state: BaseGHZState,
                           control_limits: Tuple[float, float],
                           alg="GRAPE") -> OptimResult:
    psi_0 = qutip.tensor([qutip.basis(2, 1) for _ in range(N)])

    norm_H_d, norm_H_c = get_hamiltonian_variables(N, 1, RegularLattice(geometry.shape, spacing=1.))
    target_state = ghz_state.get_state_tensor()

    optim_shared_kwargs = dict(
        amp_lbound=control_limits[0] / characteristic_V,
        amp_ubound=control_limits[1] / characteristic_V,
        gen_stats=True,
        max_wall_time=300, max_iter=10000, fid_err_targ=1e-10,
        log_level=qutip.logging_utils.WARN,
        # min_grad=1e-16,
    )

    optim_kwargs_ = {**optim_shared_kwargs}

    if alg == "GRAPE":
        optim_kwargs_['init_pulse_type'] = "RND"
    else:
        optim_kwargs_['guess_pulse_type'] = "RND"

    if alg == "GRAPE":
        norm_result = cpo.optimize_pulse_unitary(
            norm_H_d, norm_H_c,
            psi_0, target_state,
            timesteps, norm_t,

            # pulse_scaling=1e9 * norm_scaling, pulse_offset=1e9 * norm_scaling,
            # pulse_scaling=0.5,
            # optim_method="FMIN_BFGS",
            **optim_kwargs_
        )
    else:
        norm_result = cpo.opt_pulse_crab_unitary(
            norm_H_d, norm_H_c,
            psi_0, target_state,
            timesteps, norm_t,
            num_coeffs=timesteps * 2,
            # guess_pulse_scaling=0.1,
            # guess_pulse_scaling=1e9 * norm_scaling, guess_pulse_offset=1e9 * norm_scaling,
            **optim_kwargs_
        )
    return norm_result


def report_stats(result: OptimResult, ghz_state: BaseGHZState):
    result.stats.report()
    target_state = ghz_state.get_state_tensor()
    final_fidelity = qutip.fidelity(target_state, result.evo_full_final) ** 2
    print(f"Final fidelity: {final_fidelity:.5f}")

    print(f"Final gradient normal: {result.grad_norm_final:.3e}")
    print(f"Terminated due to: {result.termination_reason}")


def plot_optimresult(result: OptimResult, N: int, t: float, timesteps: int,
                     V: float, characteristic_V: float,
                     geometry: RegularLattice,
                     ghz_single_component: List[bool],
                     ):
    # Omega = np.hstack((result.final_amps[:, 0], result.final_amps[-1, 0]))
    # Delta = np.hstack((result.final_amps[:, 1], result.final_amps[-1, 1]))
    Omega = result.final_amps[:, 0] * characteristic_V
    Delta = result.final_amps[:, 1] * characteristic_V

    print(f"Omega: {Omega.tolist()}")
    print(f"Delta: {Delta.tolist()}")

    t_list = np.linspace(0, t, timesteps + 1)
    Omega_func = interp1d(t_list, np.hstack((Omega, Omega[-1])), kind="previous", bounds_error=False, fill_value=0)
    Delta_func = interp1d(t_list, np.hstack((Delta, Delta[-1])), kind="previous", bounds_error=False, fill_value=0)

    t_list = np.linspace(0, t, 3001)
    Omega = np.array([Omega_func(_t) for _t in t_list[:-1]])
    Delta = np.array([Delta_func(_t) for _t in t_list[:-1]])

    e_qs = EvolvingQubitSystem(
        N=N, V=V, geometry=geometry,
        Omega=Omega,
        Delta=Delta,
        t_list=t_list,
        ghz_state=QuimbGHZState(N, ghz_single_component),
    )
    e_qs.solve()
    print(f"Solved system fidelity: {e_qs.get_fidelity_with('ghz'):.5f}")

    save_protocol(
        f"{alg}_1D_STD_8",
        N=N, V=V, geometry=geometry, GHZ_single_component=ghz_single_component,
        Omega=Omega,
        Delta=Delta,
        t_list=t_list,
        fidelity=e_qs.get_fidelity_with('ghz')
    )

    e_qs.plot(show=True)


if __name__ == '__main__':
    import numpy as np
    from qutip_job_handlers.crossing import get_ghz_crossing
    from job_handlers.timer import timer
    import interaction_constants

    N = 8
    t = 2e-6
    timesteps = 25
    LATTICE_SPACING = 1.5e-6

    geometry = RegularLattice(shape=(8,), spacing=LATTICE_SPACING)
    ghz_single_component = [True, True, True, True, True, True, True, True]

    # geometry = RegularLattice(shape=(4, 2), spacing=LATTICE_SPACING)
    # ghz_single_component = [True, False, False, True, True, False, False, True]
    ghz_state = CustomGHZState(N, ghz_single_component)

    N_RYD = 50
    C6 = interaction_constants.get_C6(N_RYD)

    print(f"C6: {C6:.3e}")
    characteristic_V = C6 / (LATTICE_SPACING ** 6)
    print(f"Characteristic V: {characteristic_V:.3e} Hz")

    norm_t = t * characteristic_V
    print(f"norm_t {norm_t}")

    with timer(f"Creating QutipSpinHam (N={N})"):
        spin_ham = QutipSpinHamiltonian(N)

    with timer(f"Calculating crossing"):
        crossing = get_ghz_crossing(
            spin_ham=spin_ham, characteristic_V=characteristic_V,
            ghz_state=ghz_state, geometry=geometry,
            V=C6
        )

    Omega_and_Delta_limits = (0, 1.5 * crossing)
    alg = "GRAPE"
    result = get_optimised_controls(N, timesteps, norm_t, characteristic_V, geometry, ghz_state, alg=alg,
                                    control_limits=Omega_and_Delta_limits)
    report_stats(result, ghz_state)
    plot_optimresult(result, N, t, timesteps, C6, characteristic_V, geometry, ghz_single_component)
