import os

import krotov
import matplotlib.pyplot as plt
import numpy as np
from krotov.result import Result

import interaction_constants
from demonstration_utils import get_normalised_hamiltonian, solve_and_print_stats
from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.qubit_system_classes import EvolvingQubitSystem
from qubit_system.utils.ghz_states import StandardGHZState
from qubit_system.utils.interpolation import get_hamiltonian_coeff_interpolation, \
    get_hamiltonian_coeff_linear_interpolation


OPT_RESULT_DUMP = os.getenv("OPT_RESULT_DUMP", "demonstration_4_krotov_8_3_1")
CONTINUE_FROM = bool(eval(os.getenv("CONTINUE_FROM", "True")))

N = int(os.getenv("QUBIT_N", "8"))
t = float(os.getenv("QUBIT_T", "0.05e-6"))

Omega = eval(os.getenv("OMEGA", "[0, t / 5, t * 4 / 5, t], [0, 630e6, 300e6, 0]"))
Delta = eval(os.getenv("DELTA", "[0, t], [1.3e9, 1.2e9]"))

GHZ_SYMMETRIC = bool(eval(os.getenv("GHZ_SYMMETRIC", "False")))

SHOW_PLOTS = bool(eval(os.getenv("SHOW_PLOTS", "True")))


N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

LATTICE_SPACING = 1.5e-6

print(f"C6: {C6:.3e}")
characteristic_V = C6 / (LATTICE_SPACING ** 6)
print(f"Characteristic V: {characteristic_V:.3e} Hz")

norm_V = C6 / (LATTICE_SPACING ** 6) / characteristic_V


norm_Omega_t, norm_Omega = np.array(Omega[0]) * characteristic_V, np.array(Omega[1]) / characteristic_V
norm_Delta_t, norm_Delta = np.array(Delta[0]) * characteristic_V, np.array(Delta[1]) / characteristic_V

norm_t = t * characteristic_V
norm_H_d, norm_H_c, psi_0 = get_normalised_hamiltonian(N, norm_V)
target_state = StandardGHZState(N).get_state_tensor(GHZ_SYMMETRIC)


norm_t_list: np.ndarray = np.linspace(0, norm_t, 500)
H = [norm_H_d, [norm_H_c[0], lambda t, args: 1], [norm_H_c[1], lambda t, args: 0.1]]

objectives = [
    krotov.Objective(initial_state=psi_0, target=target_state, H=H)
]


def S(t):
    """Shape function for the field update"""
    return krotov.shapes.flattop(t, t_start=0, t_stop=norm_t, t_rise=norm_t / 10, t_fall=norm_t / 10, func='sinsq')


def shape_field(eps0):
    """Applies the shape function S(t) to the guess field"""
    eps0_shaped = lambda t, args: eps0(t, args)*S(t)
    return eps0_shaped


# H[1][1] = shape_field(H[1][1])
H[1][1] = get_hamiltonian_coeff_linear_interpolation(norm_Omega_t, norm_Omega)
H[2][1] = get_hamiltonian_coeff_linear_interpolation(norm_Delta_t, norm_Delta)

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


if SHOW_PLOTS and not CONTINUE_FROM:
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
if SHOW_PLOTS and not CONTINUE_FROM:
    guess_dynamics = objectives[0].mesolve(norm_t_list, e_ops=[proj_G, proj_GHZ])
    plot_population(guess_dynamics)

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
    iter_stop=40,
    continue_from=Result.load(OPT_RESULT_DUMP, objectives=objectives) if CONTINUE_FROM else None
)

dump_file_name = OPT_RESULT_DUMP if not CONTINUE_FROM else OPT_RESULT_DUMP + "_1"
opt_result.dump(dump_file_name)

print(opt_result)

if SHOW_PLOTS:
    plot_pulse(opt_result.optimized_controls[0], norm_t_list, "Optimised $\Omega$")
    plot_pulse(opt_result.optimized_controls[1], norm_t_list, "Optimised $\Delta$")

opt_dynamics = opt_result.optimized_objectives[0].mesolve(
    norm_t_list, e_ops=[proj_G, proj_GHZ])
plot_population(opt_dynamics)

final_Omega = opt_result.optimized_controls[0] * characteristic_V
final_Delta = opt_result.optimized_controls[1] * characteristic_V
t_list = norm_t_list / characteristic_V
e_qs = EvolvingQubitSystem(
    N=N, V=C6, geometry=RegularLattice1D(LATTICE_SPACING),
    Omega=get_hamiltonian_coeff_interpolation(t_list, final_Omega, "cubic"),
    Delta=get_hamiltonian_coeff_interpolation(t_list, final_Delta, "cubic"),
    t_list=np.linspace(0, t, 300),
    ghz_state=StandardGHZState(N)
)
solve_and_print_stats(e_qs)

plt.show()
