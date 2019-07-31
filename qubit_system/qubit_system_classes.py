from pathlib import Path
from typing import Callable, Optional

import numpy as np
from tqdm.auto import tqdm
from matplotlib import ticker, pyplot as plt
from qutip import qeye, Qobj, mesolve, Options, fidelity, tensor, expect, sesolve
from qutip.solver import Result

from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.utils.states import get_exp_list, get_ground_states, get_states, get_label_from_state, get_ghz_state, \
    get_excited_states
from qubit_system.utils.interpolation import get_hamiltonian_coeff_linear_interpolation

PLOT_FOLDER = Path(__file__).parent.parent / 'plots'
PLOT_FOLDER.mkdir(exist_ok=True)


def plotting_decorator(func):
    def wrapper(*args, show: bool = False, savefig_name: str = None, **kwargs):
        return_value = func(*args, **kwargs)

        if savefig_name:
            plt.savefig(PLOT_FOLDER / savefig_name, dpi=300)
        if show:
            plt.show()

        return return_value

    return wrapper


class BaseQubitSystem:
    def __init__(self, N: int, V: float, geometry: BaseGeometry):
        self.N = N
        self.V = V

        self.geometry = geometry


class StaticQubitSystem(BaseQubitSystem):
    def __init__(self, N: int, V: float, geometry: BaseGeometry, Omega: float, Delta: np.ndarray):
        super().__init__(N, V, geometry)

        self.Omega = Omega
        self.Delta = Delta

    def get_hamiltonian(self, detuning: float) -> Qobj:
        # noinspection PyTypeChecker
        H: Qobj = 0

        sx_list, sy_list, sz_list = get_exp_list(self.N)

        for i in range(self.N):
            H += self.Omega / 2 * sx_list[i]
            n_i = (sz_list[i] + qeye(1)) / 2
            H -= detuning * n_i

            for j in range(i):
                n_j = (sz_list[j] + qeye(1)) / 2

                H += self.V / self.geometry.get_distance(i, j) ** 6 * n_i * n_j
        return H

    def plot(self):
        self.plot_detuning_energy_levels(
            plot_state_names=self.N <= 4,
            savefig_name=f"detuning_{self.N}.png",
            show=True
        )

    @plotting_decorator
    def plot_detuning_energy_levels(self, plot_state_names: bool):
        states = get_states(self.N)

        plot_points = len(self.Delta)
        Omega_is_zero = self.Omega == 0

        omega_zero_all_energies = []
        omega_non_zero_all_energies = []
        for detuning in tqdm(self.Delta):
            H = self.get_hamiltonian(detuning)
            energies = []

            for state in states:
                energy = expect(H, tensor(*state))
                energies.append(energy)
            omega_zero_all_energies.append(energies)
            if not Omega_is_zero:
                eigenvalues, eigenstates = H.eigenstates()
                omega_non_zero_all_energies.append(eigenvalues)
        omega_zero_all_energies = np.array(omega_zero_all_energies)
        omega_non_zero_all_energies = np.array(omega_non_zero_all_energies)

        plt.figure(figsize=(15, 7), num="Energy Levels")

        for i in reversed(range(len(states))):
            label = get_label_from_state(states[i])
            color = 'g' if 'e' not in label else 'r' if 'g' not in label else 'grey'
            linewidth = 5 if 'e' not in label or 'g' not in label else 1
            # color = f'C{i}'
            plt.plot(self.Delta, omega_zero_all_energies[:, i], color=color, label=label, alpha=0.6, lw=linewidth)
            if not Omega_is_zero:
                plt.plot(self.Delta, omega_non_zero_all_energies[:, i], color=f'C{i}', ls=':', alpha=0.6)

            if plot_state_names:
                Delta_index = int(plot_points / len(states)) * i + int(plot_points / 2 / len(states))
                text_x = self.Delta[Delta_index]
                text_y = omega_zero_all_energies[Delta_index, i]
                plt.text(text_x, text_y, label, ha='center', color=f'C{i}', fontsize=16, fontweight='bold')

        if plot_state_names:
            plt.legend()

        plt.grid()
        ax = plt.gca()
        scaled_xaxis_ticker = ticker.EngFormatter(unit="Hz")
        scaled_yaxis_ticker = ticker.EngFormatter(unit="Hz")
        ax.xaxis.set_major_formatter(scaled_xaxis_ticker)
        ax.yaxis.set_major_formatter(scaled_yaxis_ticker)

        # plt.title(rf"Energy spectrum with $N = {self.N}$, $V = {self.V:0.2e}$, $\Omega = {self.Omega:0.2e}$")
        _m, _s = f"{self.V:0.2e}".split('e')
        V_text = rf"{_m:s} \times 10^{{{int(_s):d}}}"
        plt.title(rf"Energy spectrum with $N = {self.N}$, $V = {V_text:s}$ Hz")
        plt.xlabel(r"Detuning $\Delta$")
        plt.ylabel("Eigenenergy")
        plt.tight_layout()


class EvolvingQubitSystem(BaseQubitSystem):
    def __init__(self, N: int, V: float, geometry: BaseGeometry,
                 Omega: Callable[[float], float],
                 Delta: Callable[[float], float],
                 t_list: np.ndarray, ghz_state: Qobj, psi_0: Qobj = None):
        super().__init__(N, V, geometry)

        self.Omega = Omega
        self.Delta = Delta
        self.t_list = t_list

        self.psi_0 = tensor(get_ground_states(N) if psi_0 is None else psi_0)
        self.ghz_state = ghz_state

        self.solve_result: Optional[Result] = None

    def get_hamiltonian(self):
        sx_list, sy_list, sz_list = get_exp_list(self.N)
        time_independent_terms = 0
        Omega_coeff_terms = 0
        Delta_coeff_terms = 0

        for i in range(self.N):
            Omega_coeff_terms += 1 / 2 * sx_list[i]
            n_i = (sz_list[i] + qeye(1)) / 2
            Delta_coeff_terms -= n_i

            for j in range(i):
                n_j = (sz_list[j] + qeye(1)) / 2

                time_independent_terms += self.V / self.geometry.get_distance(i, j) ** 6 * n_i * n_j
        return [
            time_independent_terms,
            [Omega_coeff_terms, self.Omega],
            [Delta_coeff_terms, self.Delta]
        ]

    def solve(self):
        # noinspection PyTypeChecker
        self.solve_result = sesolve(
            self.get_hamiltonian(),
            self.psi_0,
            self.t_list,
            e_ops=get_exp_list(self.N)[2],
            options=Options(store_states=True, nsteps=100000)
        )

    def plot(self):
        if self.solve_result is None:
            self.plot_Omega_and_Delta(savefig_name="omega_and_delta.png", show=True)
        else:
            self.plot_Omega_and_Delta(savefig_name="omega_and_delta.png")
            self.plot_state_overlaps(savefig_name="state_overlaps.png", show=True)

    @plotting_decorator
    def plot_Omega_and_Delta(self):
        """
        Plots Omega and Delta as a function of time.
        Includes overlap with GHZ state if `self.solve_result` is not None (self.solve has been called).
        :return:
        """
        rows = 2 if self.solve_result is None else 3
        fig, axs = plt.subplots(rows, 1, sharex='all', figsize=(15, rows * 3), num="Omega and Delta")
        plt.xlabel('Time')

        ax0 = axs[0]
        ax0.plot(self.t_list, [self.Omega(t) for t in self.t_list])
        ax0.set_ylabel("Omega")
        ax0.yaxis.set_major_formatter(ticker.EngFormatter('Hz'))

        ax1 = axs[1]
        ax1.plot(self.t_list, [self.Delta(t) for t in self.t_list])
        ax1.set_ylabel("Delta")
        ax1.yaxis.set_major_formatter(ticker.EngFormatter('Hz'))

        ax0.xaxis.set_major_formatter(ticker.EngFormatter('s'))

        if self.solve_result is not None:
            ax2 = axs[2]
            ax2.plot(
                self.t_list,
                [fidelity(self.ghz_state, _instantaneous_state) ** 2
                 for _instantaneous_state in self.solve_result.states]
            )
            ax2.set_ylabel("Fidelity with GHZ")
            ax2.set_ylim((-0.1, 1.1))
            ax2.yaxis.set_ticks([0, 0.5, 1])

        for ax in axs:
            ax.grid()

        plt.tight_layout()

    @plotting_decorator
    def plot_state_overlaps(self):
        states = get_states(self.N)
        fig, axs = plt.subplots(len(states), 1, sharex='all', figsize=(15, 3 * 2 ** self.N + 3), num="State overlaps")

        for i, state in enumerate(states):
            ax = axs[i]
            ax.plot(
                self.t_list,
                [fidelity(tensor(state), _instantaneous_state) ** 2
                 for _instantaneous_state in self.solve_result.states]
            )
            label = get_label_from_state(state)
            ax.set_ylabel(label)
            ax.set_ylim((-0.1, 1.1))
            ax.yaxis.set_ticks([0, 1])
            ax.grid()

        axs[0].xaxis.set_major_formatter(ticker.EngFormatter('s'))
        plt.tight_layout()

    def get_fidelity_with(self, target_state: str = "ghz") -> float:
        """
        :param target_state: One of "ghz", "ground", and "excited"
        :return:
        """
        assert (self.solve_result is not None), "solve_result attribute cannot be None (call solve method)"
        final_state = self.solve_result.states[-1]
        if target_state == "ghz":
            return fidelity(final_state, self.ghz_state) ** 2
        elif target_state == "ground":
            return fidelity(final_state, tensor(*get_ground_states(self.N))) ** 2
        elif target_state == "excited":
            return fidelity(final_state, tensor(*get_excited_states(self.N))) ** 2
        else:
            raise ValueError(f"target_state has to be one of 'ghz', 'ground', or 'excited', not {target_state}.")


if __name__ == '__main__':
    from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
    from qubit_system.geometry.regular_lattice_2d import RegularLattice2D

    # s_qs = StaticQubitSystem(
    #     N=4, V=1,
    #     geometry=RegularLattice1D(),
    #     Omega=1, Delta=np.linspace(-1, 1, 50)
    # )
    # s_qs.plot()
    #
    # s_qs = StaticQubitSystem(
    #     N=4, V=1,
    #     geometry=RegularLattice2D((2, 2)),
    #     Omega=1, Delta=np.linspace(-1, 1, 50)
    # )
    # s_qs.plot()

    s_qs = StaticQubitSystem(
        N=2, V=1,
        geometry=RegularLattice1D(),
        Omega=1, Delta=np.linspace(-1, 1, 50)
    )
    s_qs.plot()

    t = 1
    e_qs = EvolvingQubitSystem(
        N=2, V=1, geometry=RegularLattice1D(),
        Omega=get_hamiltonian_coeff_linear_interpolation([0, t / 4, t * 3 / 4, t], [0, 1, 1, 0]),
        Delta=get_hamiltonian_coeff_linear_interpolation([0, t], [-1, 1]),
        t_list=np.linspace(0, 1, 100),
        ghz_state=get_ghz_state(2)
    )
    e_qs.solve()
    e_qs.plot()

    # t = 1
    # e_qs = EvolvingQubitSystem(
    #     N=4, V=1, geometry=RegularLattice2D((2, 2)),
    #     Omega=get_hamiltonian_coeff_linear_interpolation([0, t / 4, t * 3 / 4, t], [0, 1, 1, 0]),
    #     Delta=get_hamiltonian_coeff_linear_interpolation([0, t], [-1, 1]),
    #     t_list=np.linspace(0, 1, 100),
    #     ghz_state=get_ghz_state(4)
    # )
    # e_qs.solve()
    # e_qs.plot()
