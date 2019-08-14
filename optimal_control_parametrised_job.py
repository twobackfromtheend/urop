import os
import pickle
from pathlib import Path

import numpy as np
import qutip
import qutip.control.pulseoptim as cpo
from qutip.control.optimresult import OptimResult

import interaction_constants
from ifttt_webhook import trigger_event
from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.geometry.regular_lattice_2d import RegularLattice2D
from qubit_system.geometry.regular_lattice_3d import RegularLattice3D
from qubit_system.qubit_system_classes import StaticQubitSystem, EvolvingQubitSystem
from qubit_system.utils.ghz_states import StandardGHZState
from qubit_system.utils.interpolation import get_hamiltonian_coeff_linear_interpolation, \
    get_hamiltonian_coeff_interpolation

import matplotlib.pyplot as plt


N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

LATTICE_SPACING = 1.5e-6

print(f"C6: {C6:.3e}")
characteristic_V = C6 / (LATTICE_SPACING ** 6)
print(f"Characteristic V: {characteristic_V:.3e} Hz")

t = 2e-6


def solve_and_print_stats(e_qs: EvolvingQubitSystem, **kwargs):
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
    e_qs.plot(with_antisymmetric_ghz=True, **kwargs, fig_kwargs={'figsize': (6, 4)}, plot_titles=False,
              plot_others_as_sum=True
              )


norm_V = C6 / (LATTICE_SPACING ** 6) / characteristic_V
norm_t = t * characteristic_V


def get_normalised_hamiltonian(N: int, norm_geometry: BaseGeometry):
    norm_e_qs = EvolvingQubitSystem(
        N=N, V=norm_V, geometry=norm_geometry,
        Omega=None, Delta=None,
        t_list=None,
        ghz_state=None
    )
    norm_hamiltonian = norm_e_qs.get_hamiltonian()

    norm_H_d = norm_hamiltonian[0]  # "drift": time-independent part
    norm_H_c = [norm_hamiltonian[1][0], norm_hamiltonian[2][0]]  # "control": time-dependent parts
    return norm_H_d, norm_H_c, norm_e_qs.psi_0


def get_optimised_controls(N: int, n_ts: int, alg: str, norm_geometry: BaseGeometry) -> OptimResult:
    norm_H_d, norm_H_c, psi_0 = get_normalised_hamiltonian(N, norm_geometry)
    target_state = StandardGHZState(N).get_state_tensor()

    norm_scaling = 0.5 / characteristic_V

    optim_shared_kwargs = dict(
        amp_lbound=-10, amp_ubound=10,
        # amp_lbound=0, amp_ubound=2e9 * norm_scaling,
        gen_stats=True,
        max_wall_time=300, max_iter=100000, fid_err_targ=1e-10,
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
            num_coeffs=20,
            guess_pulse_scaling=0.1,
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


def plot_optimresult(result: OptimResult, N: int, t: float, geometry: BaseGeometry, unnormalise_V: float, **kwargs):
        time = result.time
        final_Omega = np.hstack((result.final_amps[:, 0], result.final_amps[-1, 0]))
        final_Delta = np.hstack((result.final_amps[:, 1], result.final_amps[-1, 1]))

        t /= unnormalise_V
        time = result.time / unnormalise_V
        final_Omega *= unnormalise_V
        final_Delta *= unnormalise_V

        e_qs = EvolvingQubitSystem(
            N=N, V=C6, geometry=geometry,
            Omega=get_hamiltonian_coeff_interpolation(time, final_Omega, "previous"),
            Delta=get_hamiltonian_coeff_interpolation(time, final_Delta, "previous"),
            t_list=np.linspace(0, t, 300),
            ghz_state=StandardGHZState(N)
        )
        solve_and_print_stats(e_qs, **kwargs)


N = int(os.getenv("N"))
job_id = os.getenv("PBS_JOBID")
alg = os.getenv("ALG")
geometry_envvar = eval(os.getenv("QUBIT_GEOMETRY"))
if geometry_envvar == 1:
    geometry = RegularLattice1D(LATTICE_SPACING)
    norm_geometry = RegularLattice1D()
elif len(geometry_envvar) == 2:
    geometry = RegularLattice2D(geometry_envvar, spacing=LATTICE_SPACING)
    norm_geometry = RegularLattice2D(geometry_envvar)
elif len(geometry_envvar) == 3:
    geometry = RegularLattice3D(geometry_envvar, spacing=LATTICE_SPACING)
    norm_geometry = RegularLattice3D(geometry_envvar)
else:
    raise ValueError('QUBIT_GEOMETRY has to be either "1", "(X, Y)", or "(X, Y, Z)"')

OPTIM_RESULT_FOLDER = Path(__file__).parent / 'optim_results'
OPTIM_RESULT_FOLDER.mkdir(exist_ok=True)

trigger_event("job_progress", value1="Job started", value2=job_id)

optim_result = get_optimised_controls(N, n_ts=15, alg=alg, norm_geometry=norm_geometry)
report_stats(optim_result, N)
with (OPTIM_RESULT_FOLDER / f"optim_result_{job_id}.pkl").open('wb') as f:
    pickle.dump(optim_result, f, protocol=pickle.HIGHEST_PROTOCOL)
plot_optimresult(optim_result, N, norm_t, geometry, characteristic_V, savefig_name=f"optim_result_{job_id}.png")

trigger_event("job_progress", value1="Job ended", value2=job_id)
