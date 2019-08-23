from pathlib import Path
from typing import Callable, Optional, Union, Tuple

import numpy as np
from matplotlib import ticker, pyplot as plt
from qutip import qeye, Qobj, Options, fidelity, tensor, expect, sesolve, mcsolve, essolve
from qutip.solver import Result
from tqdm.auto import tqdm

from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.utils.ghz_states import BaseGHZState
from qubit_system.utils.interpolation import get_hamiltonian_coeff_linear_interpolation
from qubit_system.utils.states import get_exp_list, get_ground_states, get_states, get_label_from_state, \
    get_excited_states, get_product_basis_states_index

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
    def plot_detuning_energy_levels(self, plot_state_names: bool, fig_kwargs: dict = None, plot_title: bool = True,
                                    ylim: Tuple[float, float] = None):
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

        if fig_kwargs is None:
            fig_kwargs = {}
        fig_kwargs = {**dict(figsize=(15, 7), num="Energy Levels"), **fig_kwargs}

        plt.figure(**fig_kwargs)

        for i in reversed(range(len(states))):
            label = get_label_from_state(states[i])
            color = 'g' if 'e' not in label else 'r' if 'g' not in label else 'grey'
            excited_or_ground = 'e' not in label or 'g' not in label
            linewidth = 5 if excited_or_ground else 1
            z_order = 2 if excited_or_ground else 1
            # color = f'C{i}'
            plt.plot(self.Delta, omega_zero_all_energies[:, i], color=color, label=label, alpha=0.6, lw=linewidth,
                     zorder=z_order)
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
        plt.locator_params(nbins=4)

        # plt.title(rf"Energy spectrum with $N = {self.N}$, $V = {self.V:0.2e}$, $\Omega = {self.Omega:0.2e}$")
        _m, _s = f"{self.V:0.2e}".split('e')
        if plot_title:
            V_text = rf"{_m:s} \times 10^{{{int(_s):d}}}"
            plt.title(rf"Energy spectrum with $N = {self.N}$, $V = {V_text:s}$ Hz")
        plt.xlabel(r"Detuning $\Delta$")
        plt.ylabel("Eigenenergy")
        plt.xlim((self.Delta.min(), self.Delta.max()))
        if ylim:
            plt.ylim(ylim)
        plt.tight_layout()


class EvolvingQubitSystem(BaseQubitSystem):
    def __init__(self, N: int, V: float, geometry: BaseGeometry,
                 Omega: Callable[[float], float],
                 Delta: Callable[[float], float],
                 t_list: np.ndarray, ghz_state: BaseGHZState, psi_0: Qobj = None):
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
            # e_ops=get_exp_list(self.N)[2],
            options=Options(store_states=True, nsteps=100000),
            # ntraj=2
        )

    @plotting_decorator
    def plot(self, with_antisymmetric_ghz: bool = False, fig_kwargs: dict = None, plot_titles: bool = True,
             plot_others_as_sum: bool = False):
        assert self.solve_result is not None, "solve_result attribute cannot be None (call solve method)"

        if fig_kwargs is None:
            fig_kwargs = {}
        fig_kwargs = {**dict(figsize=(15, 9)), **fig_kwargs}

        fig, axs = plt.subplots(3, 1, sharex='all', **fig_kwargs)
        self.plot_Omega_and_Delta(axs[0], plot_title=plot_titles)
        self.plot_ghz_states_overlaps(axs[1], with_antisymmetric_ghz, plot_title=plot_titles)
        self.plot_basis_states_overlaps(axs[2], plot_title=plot_titles, plot_others_as_sum=plot_others_as_sum)

        plt.xlabel('Time')
        plt.tight_layout()

        for ax in axs:
            ax.grid()

            # Move legend to the right
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def plot_Omega_and_Delta(self, ax, plot_title: bool = True):
        """
        Plots Omega and Delta as a function of time.
        Includes overlap with GHZ state if `self.solve_result` is not None (self.solve has been called).
        :return:
        """
        ax.xaxis.set_major_formatter(ticker.EngFormatter('s'))
        plt.xlabel('Time')

        ax.plot(self.t_list, [self.Omega(t) for t in self.t_list], label=r"$\Omega{}$", lw=3, alpha=0.8)
        ax.plot(self.t_list, [self.Delta(t) for t in self.t_list], label=r"$\Delta{}$", lw=3, alpha=0.8)
        ax.yaxis.set_major_formatter(ticker.EngFormatter('Hz'))
        ax.locator_params(nbins=4, axis='y')
        ax.locator_params(nbins=5, axis='x')

        ax.legend()
        if plot_title:
            ax.set_title("Control parameters")
        delta = (self.t_list.max() - self.t_list.min()) * 0.01
        ax.set_xlim((self.t_list.min() - delta, self.t_list.max() + delta))

    def plot_ghz_states_overlaps(self, ax, with_antisymmetric_ghz: bool, plot_title: bool = True):
        labelled_states = [(self.ghz_state.get_state_tensor(), r"$\psi_{\mathrm{GHZ}}^{\mathrm{s}}$")]
        if with_antisymmetric_ghz is not None:
            labelled_states.append(
                (self.ghz_state.get_state_tensor(symmetric=False), r"$\psi_{\mathrm{GHZ}}^{\mathrm{a}}$"))

        for _state, _label in labelled_states:
            ax.plot(
                self.t_list,
                [fidelity(_state, _instantaneous_state) ** 2
                 for _instantaneous_state in self.solve_result.states],
                label=_label,
                lw=1,
                alpha=0.8
            )
        ax.set_ylabel("Fidelity")
        if plot_title:
            ax.set_title("Fidelity with GHZ states")
        ax.set_ylim((-0.1, 1.1))
        ax.yaxis.set_ticks([0, 0.5, 1])
        ax.legend()

    def plot_basis_states_overlaps(self, ax, plot_title: bool = True, plot_others_as_sum: bool = False):
        states = get_states(self.N) if not plot_others_as_sum else [get_excited_states(self.N),
                                                                    get_ground_states(self.N)]
        fidelities = []

        plot_individual_orthogonal_state_labels = len(states) <= 4
        plotted_others = False
        for i, state in enumerate(tqdm(states)):
            label = get_label_from_state(state)
            state_product_basis_index = get_product_basis_states_index(state)
            # state_fidelities = [fidelity(tensor(state), _instantaneous_state) ** 2
            #                     for _instantaneous_state in self.solve_result.states]
            state_fidelities = [np.abs(_instantaneous_state.data.toarray().flatten()[state_product_basis_index]) ** 2
                                for _instantaneous_state in self.solve_result.states]

            if 'e' not in label or 'g' not in label:
                fidelities.append(state_fidelities)

            plot_label = r"$P_{" + f"{label.upper()[0]}" + "}$" \
                if plot_individual_orthogonal_state_labels or ('e' not in label or 'g' not in label) else 'Others'
            if plot_label == 'Others':
                if plotted_others:
                    plot_label = None
                else:
                    plotted_others = True

            ax.plot(
                self.t_list,
                state_fidelities,
                label=plot_label,
                color='g' if 'e' not in label else 'r' if 'g' not in label else 'k',
                linewidth=1 if 'e' not in label or 'g' not in label else 0.5,
                alpha=0.5
            )

        fidelities_sum = np.array(fidelities).sum(axis=0)
        ax.plot(self.t_list, fidelities_sum,
                label="$P_{E} + P_{G}$",
                color='C0', linestyle=":", linewidth=1, alpha=0.7)

        if plot_others_as_sum:
            others_sum = 1 - fidelities_sum

            ax.plot(self.t_list, others_sum,
                    label=r"$\sum{\textrm{Others}}$",
                    color='C1', linestyle=":", linewidth=1, alpha=0.7)

        ax.set_ylabel("Fidelity")
        if plot_title:
            ax.set_title("Fidelity with basis states")
        ax.set_ylim((-0.1, 1.1))
        ax.yaxis.set_ticks([0, 0.5, 1])

        # ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles)))
        ax.legend(handles, labels)

    def get_fidelity_with(self, target_state: Union[str, Qobj] = "ghz") -> float:
        """
        :param target_state: One of "ghz", "ghz_antisymmetric", "ground", and "excited"
        :return:
        """
        assert (self.solve_result is not None), "solve_result attribute cannot be None (call solve method)"
        final_state = self.solve_result.states[-1]
        if target_state == "ghz":
            return fidelity(final_state, self.ghz_state.get_state_tensor(symmetric=True)) ** 2
        elif target_state == "ghz_antisymmetric":
            return fidelity(final_state, self.ghz_state.get_state_tensor(symmetric=False)) ** 2
        elif target_state == "ground":
            return fidelity(final_state, tensor(*get_ground_states(self.N))) ** 2
        elif target_state == "excited":
            return fidelity(final_state, tensor(*get_excited_states(self.N))) ** 2
        elif isinstance(target_state, Qobj):
            return fidelity(final_state, target_state) ** 2
        else:
            raise ValueError(f"target_state has to be one of 'ghz', 'ground', or 'excited', not {target_state}.")


class TimeIndependentEvolvingQubitSystem(EvolvingQubitSystem):

    def __init__(self, N: int, V: float, geometry: BaseGeometry, Omega: float,
                 Delta: float, t_list: np.ndarray, ghz_state: BaseGHZState, psi_0: Qobj = None):
        Omega_func = lambda x: Omega
        Delta_func = lambda x: Delta

        super().__init__(N, V, geometry, Omega=Omega_func, Delta=Delta_func, t_list=t_list, ghz_state=ghz_state,
                         psi_0=psi_0)

        self._Omega = Omega
        self._Delta = Delta

    def get_hamiltonian(self) -> Qobj:
        sx_list, sy_list, sz_list = get_exp_list(self.N)
        H: Qobj = 0

        for i in range(self.N):
            H += self._Omega / 2 * sx_list[i]
            n_i = (sz_list[i] + qeye(1)) / 2
            H -= self._Delta * n_i

            for j in range(i):
                n_j = (sz_list[j] + qeye(1)) / 2

                H += self.V / self.geometry.get_distance(i, j) ** 6 * n_i * n_j
        return H


if __name__ == '__main__':
    from qubit_system.geometry.regular_lattice_1d import RegularLattice1D

    s_qs = StaticQubitSystem(
        N=4, V=1,
        geometry=RegularLattice1D(),
        Omega=1, Delta=np.linspace(-1, 1, 50)
    )
    s_qs.plot()

    from qubit_system.utils.ghz_states import StandardGHZState

    t = 1
    N = 4
    e_qs = EvolvingQubitSystem(
        N=N, V=1, geometry=RegularLattice1D(),
        Omega=get_hamiltonian_coeff_linear_interpolation([0, t / 4, t * 3 / 4, t], [0, 1, 1, 0]),
        Delta=get_hamiltonian_coeff_linear_interpolation([0, t], [-1, 1]),
        t_list=np.linspace(0, 1, 100),
        ghz_state=StandardGHZState(N)
    )
    e_qs.solve()
    e_qs.plot(with_antisymmetric_ghz=True, show=True)
