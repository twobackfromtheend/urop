import os
from typing import Callable

import krotov
import matplotlib.pyplot as plt
import numpy as np
import qutip
from scipy import interpolate

import interaction_constants
from optimised_protocols.saver import save_protocol
from qubit_system.geometry import RegularLattice
from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem
from qutip_job_handlers.hamiltonian import QutipSpinHamiltonian
from qutip_job_handlers.qutip_optim import get_hamiltonian_variables
from qutip_job_handlers.qutip_states import CustomGHZState
from qubit_system.utils.ghz_states import CustomGHZState as QuimbGHZState


def get_hamiltonian_coeff_interpolation(x: np.array, y: np.array, kind: str) -> Callable[[float], float]:
    """
    Clips to 0 beyond x range.
    :param x:
    :param y:
    :return: Function to use as Hamiltonian coefficient
    """
    f = interpolate.interp1d(x, y, kind=kind)

    def coeff_fn(t: float, args: dict = None):
        if t < min(x) or t > max(x):
            return 0
        return f(t)

    return coeff_fn


CONTINUE_FROM = False

import numpy as np
from qutip_job_handlers.crossing import get_ghz_crossing
from job_handlers.timer import timer
import interaction_constants

N = 8
t = 1e-6
timesteps = 8
LATTICE_SPACING = 1.5e-6

geometry = RegularLattice(shape=(8,), spacing=LATTICE_SPACING)
ghz_single_component = [True, True, True, True, True, True, True, True]
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

SHOW_PLOTS = True

norm_t = t * characteristic_V
psi_0 = qutip.tensor([qutip.basis(2, 1) for _ in range(N)])

norm_H_d, norm_H_c = get_hamiltonian_variables(N, 1, RegularLattice(geometry.shape, spacing=1.))
target_state = ghz_state.get_state_tensor()

timesteps = 3000
norm_t_list: np.ndarray = np.linspace(0, norm_t, timesteps)
H = [norm_H_d, [norm_H_c[0], lambda t, args: 1], [norm_H_c[1], lambda t, args: 0.1]]

objectives = [
    krotov.Objective(initial_state=psi_0, target=target_state, H=H)
]


def S(t):
    """Shape function for the field update"""
    return krotov.shapes.flattop(t, t_start=0, t_stop=norm_t, t_rise=norm_t / 10, t_fall=norm_t / 10, func='sinsq')


def shape_field(eps0):
    """Applies the shape function S(t) to the guess field"""
    eps0_shaped = lambda t, args: eps0(t, args) * S(t)
    return eps0_shaped


Omega = [0, t / 3, t * 2 / 3, t], [0, 1e9, 1e9, 0]
Delta = [0, t], [1.2e9, 1.215e9]
norm_Omega_t, norm_Omega = np.array(Omega[0]) * characteristic_V, np.array(Omega[1]) / characteristic_V
norm_Delta_t, norm_Delta = np.array(Delta[0]) * characteristic_V, np.array(Delta[1]) / characteristic_V
# H[1][1] = shape_field(H[1][1])
H[1][1] = get_hamiltonian_coeff_interpolation(norm_Omega_t, norm_Omega, kind="linear")
H[2][1] = get_hamiltonian_coeff_interpolation(norm_Delta_t, norm_Delta, kind="linear")

pulse_options = {
    H[1][1]: dict(lambda_a=20, shape=1),
    H[2][1]: dict(lambda_a=20, shape=1)
}


def print_fidelity(**args):
    F_re = np.average(np.array(args['tau_vals']).real)
    print("    F = %f" % F_re)
    return F_re


def plot_pulse(pulse, _t_list: np.ndarray, label: str):
    fig, ax = plt.subplots()
    if callable(pulse):
        pulse = np.array([pulse(t, args=None) for t in _t_list])
    ax.plot(_t_list, pulse)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'Pulse Amplitude: {label}')
    plt.show()


plot_pulse(H[1][1], norm_t_list, "$\Omega$")
plot_pulse(H[2][1], norm_t_list, "$\Delta$")


def plot_population(result):
    fig, ax = plt.subplots()
    ax.plot(result.times, result.expect[0], label='$P_{\mathrm{G}}$')
    ax.plot(result.times, result.expect[1], label='$P_{\mathrm{GHZ}}$')
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Population')
    plt.show()


proj_G = psi_0 * psi_0.dag()
proj_GHZ = target_state * target_state.dag()

guess_dynamics = objectives[0].mesolve(norm_t_list, e_ops=[proj_G, proj_GHZ])
plot_population(guess_dynamics)
print("OPTIMISING")

opt_result = krotov.optimize_pulses(
    objectives,
    pulse_options=pulse_options,
    tlist=norm_t_list,
    propagator=krotov.propagators.expm,
    chi_constructor=krotov.functionals.chis_re,
    # info_hook=krotov.info_hooks.print_table(J_T=krotov.functionals.J_T_re),
    # info_hook=krotov.info_hooks.chain(krotov.info_hooks.print_debug_information, print_fidelity),
    info_hook=print_fidelity,
    # check_convergence=krotov.convergence.check_monotonic_error,
    iter_stop=2,
    continue_from=None
)

# dump_file_name = OPT_RESULT_DUMP if not CONTINUE_FROM else OPT_RESULT_DUMP + "_1"
# opt_result.dump(dump_file_name)

print(opt_result)

plot_pulse(opt_result.optimized_controls[0], norm_t_list, "Optimised $\Omega$")
plot_pulse(opt_result.optimized_controls[1], norm_t_list, "Optimised $\Delta$")

opt_dynamics = opt_result.optimized_objectives[0].mesolve(
    norm_t_list, e_ops=[proj_G, proj_GHZ])
plot_population(opt_dynamics)


final_Omega = opt_result.optimized_controls[0] * characteristic_V
final_Delta = opt_result.optimized_controls[1] * characteristic_V
t_list = np.linspace(0, t, timesteps + 1)

e_qs = EvolvingQubitSystem(
    N=N, V=C6, geometry=geometry,
    Omega=final_Omega,
    Delta=final_Delta,
    t_list=t_list,
    ghz_state=QuimbGHZState(N, ghz_single_component)
)

e_qs.solve()
print(f"Solved system fidelity: {e_qs.get_fidelity_with('ghz'):.5f}")

save_protocol(
    "KROTOV_1D_STD_8",
    N=N, V=C6, geometry=geometry, GHZ_single_component=ghz_single_component,
    Omega=final_Omega,
    Delta=final_Delta,
    t_list=t_list,
    fidelity=e_qs.get_fidelity_with('ghz')
)

e_qs.plot(show=True)

plt.show()
