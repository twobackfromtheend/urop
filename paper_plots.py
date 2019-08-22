import pickle

import numpy as np
import qutip
import qutip.control.pulseoptim as cpo
from qutip.control.optimresult import OptimResult

import interaction_constants
from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.geometry.regular_lattice_2d import RegularLattice2D
from qubit_system.geometry.regular_lattice_3d import RegularLattice3D
from qubit_system.qubit_system_classes import StaticQubitSystem, EvolvingQubitSystem
from qubit_system.utils.ghz_states import StandardGHZState
from qubit_system.utils.interpolation import get_hamiltonian_coeff_linear_interpolation, \
    get_hamiltonian_coeff_interpolation

import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family="serif", serif="CMU Serif")
plt.rc('text.latex', preamble=r'\usepackage{upgreek}')

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

LATTICE_SPACING = 1.5e-6

print(f"C6: {C6:.3e}")
characteristic_V = C6 / (LATTICE_SPACING ** 6)
print(f"Characteristic V: {characteristic_V:.3e} Hz")

t = 2e-6


def generate_plots_1():
    """
    Generates energy level diagrams for V > 0, V < 0, and 1D, 2D, 3D,
    """
    N = 12

    for C6_coeff in [-1]:
        _C6 = C6 * C6_coeff
        for _geometry in [RegularLattice1D(spacing=LATTICE_SPACING), RegularLattice2D((3, 4), spacing=LATTICE_SPACING),
                          RegularLattice3D((2, 2, 3), spacing=LATTICE_SPACING)]:
            s_qs = StaticQubitSystem(
                N=N, V=_C6,
                geometry=_geometry,
                Omega=0, Delta=np.linspace(-2e9, 4e9, 2) * C6_coeff
            )

            V_sign = "V-" if _C6 < 0 else "V+"
            _geometry_str = str(_geometry.shape) if not isinstance(_geometry, RegularLattice1D) else "1D"
            s_qs.plot_detuning_energy_levels(False, savefig_name=f"paper_plots_1_{V_sign}_shape_{_geometry_str}.png",
                                             fig_kwargs={'figsize': (5, 4)}, plot_title=False, ylim=(-20e9, 60e9))
            plt.clf()


def generate_plots_2():
    """
    Generates plots for evolving systems for different optimisation methods
    """

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

    def get_normalised_hamiltonian(N: int):
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

    def get_optimised_controls(N: int, n_ts: int, alg: str) -> OptimResult:
        norm_H_d, norm_H_c, psi_0 = get_normalised_hamiltonian(N)
        target_state = StandardGHZState(N).get_state_tensor()

        norm_scaling = 0.5 / characteristic_V

        optim_shared_kwargs = dict(
            amp_lbound=-10, amp_ubound=10,
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
                num_coeffs=10,
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

    def plot_optimresult(result: OptimResult, N: int, geometry: BaseGeometry, t: float, unnormalise_V: float,
                         **kwargs):
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

    # N = 4
    # Manual optimisation
    # e_qs = EvolvingQubitSystem(
    #     N=N, V=C6, geometry=RegularLattice1D(spacing=LATTICE_SPACING),
    #     Omega=get_hamiltonian_coeff_linear_interpolation([0, t / 2, t], [0, 1.4148e9, 0]),
    #     Delta=get_hamiltonian_coeff_linear_interpolation([0, t], [1.5e9, 1e9]),
    #     t_list=np.linspace(0, t, 300),
    #     ghz_state=StandardGHZState(N)
    # )
    # solve_and_print_stats(e_qs, savefig_name=f'paper_plots_2_n_{N}_manual.png')

    # RL
    # with open("reinforcement_learning/results/20190814_035606.pkl", "rb") as f:
    #     data = pickle.load(f)
    # t_list = data['evolving_qubit_system_kwargs']['t_list']
    # solve_t_list = np.linspace(t_list[0], t_list[-1], 300)
    #
    # data['evolving_qubit_system_kwargs'].pop('t_list')
    # e_qs = EvolvingQubitSystem(
    #     **data['evolving_qubit_system_kwargs'],
    #     Omega=get_hamiltonian_coeff_linear_interpolation(t_list, data['protocol'].Omega),
    #     Delta=get_hamiltonian_coeff_linear_interpolation(t_list, data['protocol'].Delta),
    #     t_list=solve_t_list,
    # )
    # solve_and_print_stats(e_qs, savefig_name=f'paper_plots_2_n_{N}_RL.png')

    # GRAPE
    # optim_result = get_optimised_controls(N, n_ts=15, alg="GRAPE")
    # report_stats(optim_result, N)
    # plot_optimresult(optim_result, N, norm_t, characteristic_V, savefig_name=f'paper_plots_2_n_{N}_GRAPE.png')

    # CRAB
    # optim_result = get_optimised_controls(N, n_ts=15, alg="CRAB")
    # report_stats(optim_result, N)
    # plot_optimresult(optim_result, N, norm_t, characteristic_V, savefig_name=f'paper_plots_2_n_{N}_CRAB.png')

    N = 8  # 1d
    # Manual optimisation
    # e_qs = EvolvingQubitSystem(
    #     N=N, V=C6, geometry=RegularLattice1D(spacing=LATTICE_SPACING),
    #     Omega=get_hamiltonian_coeff_linear_interpolation([0, t / 6, t * 5 / 6, t], [0, 379e6, 379e6, 0]),
    #     Delta=get_hamiltonian_coeff_linear_interpolation([0, t], [1.3e9, 1.21e9]),
    #     t_list=np.linspace(0, t, 300),
    #     ghz_state=StandardGHZState(N)
    # )
    # solve_and_print_stats(e_qs, savefig_name=f'paper_plots_2_n_{N}_manual.png')

    # RL
    # with open("reinforcement_learning/results/20190814_211310.pkl", "rb") as f:
    #     data = pickle.load(f)
    # t_list = data['evolving_qubit_system_kwargs']['t_list']
    # solve_t_list = np.linspace(t_list[0], t_list[-1], 300)
    #
    # data['evolving_qubit_system_kwargs'].pop('t_list')
    # e_qs = EvolvingQubitSystem(
    #     **data['evolving_qubit_system_kwargs'],
    #     Omega=get_hamiltonian_coeff_linear_interpolation(t_list, data['protocol'].Omega),
    #     Delta=get_hamiltonian_coeff_linear_interpolation(t_list, data['protocol'].Delta),
    #     t_list=solve_t_list,
    # )
    # solve_and_print_stats(e_qs, savefig_name=f'paper_plots_2_n_{N}_RL.png')

    # GRAPE
    # with open("optim_results/optim_result_464779.pbs.pkl", "rb") as f:
    #     optim_result = pickle.load(f)
    # report_stats(optim_result, N)
    # plot_optimresult(optim_result, N, RegularLattice1D(LATTICE_SPACING), norm_t, characteristic_V,
    #                  savefig_name=f'paper_plots_2_n_{N}_GRAPE.png')

    # CRAB
    # with open("optim_results/optim_result_464783.pbs.pkl", "rb") as f:
    #     optim_result = pickle.load(f)
    # report_stats(optim_result, N)
    # plot_optimresult(optim_result, N, RegularLattice1D(LATTICE_SPACING), norm_t, characteristic_V,
    #                  savefig_name=f'paper_plots_2_n_{N}_CRAB.png')



    # N = 8  # 2d
    # Manual optimisation
    # e_qs = EvolvingQubitSystem(
    #     N=N, V=C6, geometry=RegularLattice2D((2, 4), spacing=LATTICE_SPACING),
    #     Omega=get_hamiltonian_coeff_linear_interpolation([0, t / 5, t * 4 / 5, t], [0, 700e6, 700e6, 0]),
    #     Delta=get_hamiltonian_coeff_linear_interpolation([0, t], [2e9, 1.83e9]),
    #     t_list=np.linspace(0, t, 300),
    #     ghz_state=StandardGHZState(N)
    # )
    # solve_and_print_stats(e_qs, savefig_name=f'paper_plots_2_n_{N}_manual_2d.png')

    # RL
    # with open("reinforcement_learning/results/20190814_211949.pkl", "rb") as f:
    #     data = pickle.load(f)
    # t_list = data['evolving_qubit_system_kwargs']['t_list']
    # solve_t_list = np.linspace(t_list[0], t_list[-1], 300)
    #
    # data['evolving_qubit_system_kwargs'].pop('t_list')
    # e_qs = EvolvingQubitSystem(
    #     **data['evolving_qubit_system_kwargs'],
    #     Omega=get_hamiltonian_coeff_linear_interpolation(t_list, data['protocol'].Omega),
    #     Delta=get_hamiltonian_coeff_linear_interpolation(t_list, data['protocol'].Delta),
    #     t_list=solve_t_list,
    # )
    # solve_and_print_stats(e_qs, savefig_name=f'paper_plots_2_n_{N}_RL_2d.png')

    # GRAPE
    # with open("optim_results/optim_result_464780.pbs.pkl", "rb") as f:
    #     optim_result = pickle.load(f)
    # report_stats(optim_result, N)
    # plot_optimresult(optim_result, N, RegularLattice2D((2, 4), LATTICE_SPACING), norm_t, characteristic_V,
    #                  savefig_name=f'paper_plots_2_n_{N}_GRAPE_2d.png')

    # CRAB
    # with open("optim_results/optim_result_464784.pbs.pkl", "rb") as f:
    #     optim_result = pickle.load(f)
    # report_stats(optim_result, N)
    # plot_optimresult(optim_result, N, RegularLattice2D((2, 4), LATTICE_SPACING), norm_t, characteristic_V,
    #                  savefig_name=f'paper_plots_2_n_{N}_CRAB_2d.png')

    N = 8  # 3d
    # Manual optimisation
    # e_qs = EvolvingQubitSystem(
    #     N=N, V=C6, geometry=RegularLattice3D((2, 2, 2), spacing=LATTICE_SPACING),
    #     Omega=get_hamiltonian_coeff_linear_interpolation([0, t / 2, t], [0, 1175e6, 0]),
    #     Delta=get_hamiltonian_coeff_linear_interpolation([0, t], [2.5e9, 2.2e9]),
    #     t_list=np.linspace(0, t, 300),
    #     ghz_state=StandardGHZState(N)
    # )
    # solve_and_print_stats(e_qs, savefig_name=f'paper_plots_2_n_{N}_manual_3d.png')

    # RL
    with open("reinforcement_learning/results/20190814_215943.pkl", "rb") as f:
        data = pickle.load(f)
    t_list = data['evolving_qubit_system_kwargs']['t_list']
    solve_t_list = np.linspace(t_list[0], t_list[-1], 300)

    data['evolving_qubit_system_kwargs'].pop('t_list')
    e_qs = EvolvingQubitSystem(
        **data['evolving_qubit_system_kwargs'],
        Omega=get_hamiltonian_coeff_linear_interpolation(t_list, data['protocol'].Omega),
        Delta=get_hamiltonian_coeff_linear_interpolation(t_list, data['protocol'].Delta),
        t_list=solve_t_list,
    )
    solve_and_print_stats(e_qs, savefig_name=f'paper_plots_2_n_{N}_RL_3d.png')

    # GRAPE
    # with open("optim_results/optim_result_464781.pbs.pkl", "rb") as f:
    #     optim_result = pickle.load(f)
    # report_stats(optim_result, N)
    # plot_optimresult(optim_result, N, RegularLattice3D((2, 2, 2), LATTICE_SPACING), norm_t, characteristic_V,
    #                  savefig_name=f'paper_plots_2_n_{N}_GRAPE_3d.png')

    # CRAB
    # with open("optim_results/optim_result_464785.pbs.pkl", "rb") as f:
    #     optim_result = pickle.load(f)
    # report_stats(optim_result, N)
    # plot_optimresult(optim_result, N, RegularLattice3D((2, 2, 2), LATTICE_SPACING), norm_t, characteristic_V,
    #                  savefig_name=f'paper_plots_2_n_{N}_CRAB_3d.png')


if __name__ == '__main__':
    # generate_plots_1()
    generate_plots_2()
