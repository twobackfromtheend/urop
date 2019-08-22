import numpy as np
import qutip
import qutip.control.pulseoptim as cpo
from qutip.control.optimresult import OptimResult

from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.qubit_system_classes import EvolvingQubitSystem
from qubit_system.utils.ghz_states import StandardGHZState
from qubit_system.utils.interpolation import get_hamiltonian_coeff_interpolation


def solve_and_print_stats(e_qs: EvolvingQubitSystem):
    import time
    start_time = time.time()
    e_qs.solve()
    print(f"Solved in {time.time() - start_time:.2f}s")

    fidelity_with_ghz = e_qs.get_fidelity_with("ghz")
    fidelity_with_ghz_asymmetric = e_qs.get_fidelity_with("ghz_antisymmetric")
    print(f"fidelity with GHZ: {fidelity_with_ghz:.4f} (with antisymmetric: {fidelity_with_ghz_asymmetric:.4f})")
    fidelity_with_ground = e_qs.get_fidelity_with("ground")
    fidelity_with_excited = e_qs.get_fidelity_with("excited")
    superposition_probability = fidelity_with_ground + fidelity_with_excited
    print(
        f"superposition probability: {superposition_probability:.4f} (g: {fidelity_with_ground:.4f}, e: {fidelity_with_excited:.4f})")
    e_qs.plot(with_antisymmetric_ghz=True)


# Normalised system for GRAPE/CRAB


def get_normalised_hamiltonian(N: int, norm_V: float):
    norm_e_qs = EvolvingQubitSystem(
        N=N, V=norm_V, geometry=RegularLattice1D(),
        Omega=None, Delta=None,
        t_list=None,
        ghz_state=None
    )
    norm_hamiltonian = norm_e_qs.get_hamiltonian()

    norm_H_d = norm_hamiltonian[0]  # "drift": time-independent part
    norm_H_c = [norm_hamiltonian[1][0], norm_hamiltonian[2][0]]  # "control": time-dependent parts
    return norm_H_d, norm_H_c, norm_e_qs.psi_0


def get_optimised_controls(N: int, n_ts: int, norm_t: float, norm_V: float, alg="GRAPE") -> OptimResult:
    norm_H_d, norm_H_c, psi_0 = get_normalised_hamiltonian(N, norm_V)
    target_state = StandardGHZState(N).get_state_tensor()

    optim_shared_kwargs = dict(
        amp_lbound=0, amp_ubound=3,
        # amp_lbound=0, amp_ubound=2e9 * norm_scaling,
        gen_stats=True,
        max_wall_time=300, max_iter=10000, fid_err_targ=1e-10,
        log_level=qutip.logging_utils.WARN,
    )
    if alg == "GRAPE":
        norm_result = cpo.optimize_pulse_unitary(
            norm_H_d, norm_H_c,
            psi_0, target_state,
            n_ts, norm_t,
            # pulse_scaling=1e9 * norm_scaling, pulse_offset=1e9 * norm_scaling,
            # pulse_scaling=0.5,
            # optim_method="FMIN_BFGS",
            init_pulse_type="RND",
            **optim_shared_kwargs
        )
    else:
        norm_result = cpo.opt_pulse_crab_unitary(
            norm_H_d, norm_H_c,
            psi_0, target_state,
            n_ts, norm_t,
            # num_coeffs=10,
            # guess_pulse_scaling=0.1,
            # guess_pulse_scaling=1e9 * norm_scaling, guess_pulse_offset=1e9 * norm_scaling,
            guess_pulse_type="RND",
            **optim_shared_kwargs
        )
    return norm_result


def report_stats(result: OptimResult, N: int):
    result.stats.report()
    target_state = StandardGHZState(N).get_state_tensor()

    final_fidelity = qutip.fidelity(target_state, result.evo_full_final) ** 2
    print(f"final_fidelity: {final_fidelity:.5f}")

    print(f"Final gradient normal {result.grad_norm_final:.3e}")
    print(f"Terminated due to {result.termination_reason}")


def plot_optimresult(result: OptimResult, N: int, t: float, C6: float, characteristic_V: float, geometry: BaseGeometry):
    final_Omega = np.hstack((result.final_amps[:, 0], result.final_amps[-1, 0]))
    final_Delta = np.hstack((result.final_amps[:, 1], result.final_amps[-1, 1]))
    time = result.time / characteristic_V
    final_Omega *= characteristic_V
    final_Delta *= characteristic_V

    e_qs = EvolvingQubitSystem(
        N=N, V=C6, geometry=geometry,
        Omega=get_hamiltonian_coeff_interpolation(time, final_Omega, "previous"),
        Delta=get_hamiltonian_coeff_interpolation(time, final_Delta, "previous"),
        t_list=np.linspace(0, t, 300),
        ghz_state=StandardGHZState(N)
    )
    solve_and_print_stats(e_qs)


__all__ = ['solve_and_print_stats', 'get_normalised_hamiltonian', 'get_optimised_controls', 'report_stats',
           'plot_optimresult']
