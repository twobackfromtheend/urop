import logging
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict

import numpy as np
import quimb as q
from matplotlib import ticker, pyplot as plt
from scipy import interpolate
from tqdm.auto import tqdm

from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.utils import states_quimb
from qubit_system.utils.ghz_states_quimb import BaseGHZState

PLOT_FOLDER = Path(__file__).parent.parent / 'plots'
PLOT_FOLDER.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)


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

    def _get_hamiltonian(self, Omega: float, Delta: float) -> q.qarray:
        sx = q.pauli("X", sparse=True)
        sz = q.pauli("Z", sparse=True)
        qnum = (sz + q.identity(2, sparse=True)) / 2
        dims = [2] * self.N

        # noinspection PyTypeChecker
        H: q.qarray = 0

        for i in range(self.N):
            H += Omega / 2 * q.ikron(sx, dims=dims, inds=i, sparse=True)
            n_i = q.ikron(qnum, dims=dims, inds=i, sparse=True)
            H -= Delta * n_i

            for j in range(i):
                n_j = q.ikron(qnum, dims=dims, inds=j, sparse=True)

                H += self.V / self.geometry.get_distance(i, j) ** 6 * n_i * n_j
        return H


class StaticQubitSystem(BaseQubitSystem):
    def __init__(self, N: int, V: float, geometry: BaseGeometry, Omega: float, Delta: np.ndarray):
        super().__init__(N, V, geometry)

        self.Omega = Omega
        self.Delta = Delta

    def get_hamiltonian(self, detuning: float) -> q.qarray:
        return self._get_hamiltonian(self.Omega, detuning)

    def plot(self):
        self.plot_detuning_energy_levels(
            plot_state_names=self.N <= 4,
            savefig_name=f"detuning_{self.N}.png",
            show=True
        )

    @plotting_decorator
    def plot_detuning_energy_levels(self, plot_state_names: bool, fig_kwargs: dict = None, plot_title: bool = True,
                                    ylim: Tuple[float, float] = None, highlight_states_by_label: List[str] = None):
        if highlight_states_by_label is None:
            highlight_states_by_label = ''.join(['e' for _ in range(self.N)])

        states = states_quimb.get_states(self.N)

        plot_points = len(self.Delta)
        Omega_is_zero = self.Omega == 0

        omega_zero_all_energies = []
        omega_non_zero_all_energies = []
        for detuning in tqdm(self.Delta):
            H = self.get_hamiltonian(detuning)
            energies = []

            for state in states:
                energy = q.expec(H, q.kron(*state)).real
                energies.append(energy)
            omega_zero_all_energies.append(energies)
            if not Omega_is_zero:
                eigenvalues = q.eigvals(H).real
                omega_non_zero_all_energies.append(eigenvalues)
        omega_zero_all_energies = np.array(omega_zero_all_energies)
        omega_non_zero_all_energies = np.array(omega_non_zero_all_energies)

        if fig_kwargs is None:
            fig_kwargs = {}
        fig_kwargs = {**dict(figsize=(15, 7), num="Energy Levels"), **fig_kwargs}

        plt.figure(**fig_kwargs)

        for i in reversed(range(len(states))):
            label = states_quimb.get_label_from_state(states[i])
            is_highlight_state = label in highlight_states_by_label
            is_ground_state = 'e' not in label
            color = 'g' if is_ground_state else 'r' if is_highlight_state else 'grey'
            linewidth = 5 if is_ground_state or is_highlight_state else 1
            z_order = 2 if is_ground_state or is_highlight_state else 1
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


class TimeIndependentEvolvingQubitSystem(BaseQubitSystem):
    def __init__(self, N: int, V: float, geometry: BaseGeometry,
                 Omega: float, Delta: float,
                 t_list: np.ndarray, ghz_state: BaseGHZState, psi_0: q.qarray = None):
        super().__init__(N, V, geometry)

        self.Omega = Omega
        self.Delta = Delta
        self.t_list = t_list

        self.psi_0 = q.kron(*states_quimb.get_ground_states(N)) if psi_0 is None else psi_0
        self.ghz_state = ghz_state

        self.evo: Optional[q.Evolution] = None
        self.solved_states: List[q.qarray] = []

    def get_hamiltonian(self) -> q.qarray:
        return self._get_hamiltonian(self.Omega, self.Delta)

    def _get_evo(self):
        return q.Evolution(
            self.psi_0,
            self.get_hamiltonian()
        )

    def solve(self) -> List[q.qarray]:
        self.evo = self._get_evo()
        self.solved_states = list(self.evo.at_times(self.t_list))
        return self.solved_states

    @plotting_decorator
    def plot(self, with_antisymmetric_ghz: bool = False, fig_kwargs: dict = None, plot_titles: bool = True,
             plot_others_as_sum: bool = False):
        assert self.evo is not None, "evo attribute cannot be None (call solve method)"

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

        ax.plot(self.t_list, [self.Omega for t in self.t_list], label=r"$\Omega{}$", lw=3, alpha=0.8)
        ax.plot(self.t_list, [self.Delta for t in self.t_list], label=r"$\Delta{}$", lw=3, alpha=0.8)
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
                [q.fidelity(_state, _instantaneous_state)
                 for _instantaneous_state in self.solved_states],
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
        states = states_quimb.get_states(self.N) if not plot_others_as_sum \
            else [states_quimb.get_excited_states(self.N), states_quimb.get_ground_states(self.N)]
        fidelities = []

        plot_individual_orthogonal_state_labels = len(states) <= 4
        plotted_others = False
        for i, state in enumerate(tqdm(states)):
            label = states_quimb.get_label_from_state(state)
            state_product_basis_index = states_quimb.get_product_basis_states_index(state)
            state_fidelities = [np.abs(_instantaneous_state.flatten()[state_product_basis_index]) ** 2
                                for _instantaneous_state in self.solved_states]

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

    def get_fidelity_with(self, target_state: Union[str, q.qarray] = "ghz") -> float:
        """
        :param target_state: One of "ghz", "ghz_antisymmetric", "ground", and "excited"
        :return:
        """
        assert (self.evo is not None), "evo attribute cannot be None (call solve method)"

        final_state = self.solved_states[-1]
        if target_state == "ghz":
            return q.fidelity(final_state, self.ghz_state.get_state_tensor(symmetric=True))
        elif target_state == "ghz_antisymmetric":
            return q.fidelity(final_state, self.ghz_state.get_state_tensor(symmetric=False))
        elif target_state == "ground":
            return q.fidelity(final_state, q.kron(*states_quimb.get_ground_states(self.N)))
        elif target_state == "excited":
            return q.fidelity(final_state, q.kron(*states_quimb.get_excited_states(self.N)))
        elif isinstance(target_state, q.qarray):
            return q.fidelity(final_state, target_state)
        else:
            raise ValueError(f"target_state has to be one of 'ghz', 'ground', or 'excited', not {target_state}.")


cached_hamiltonian_variables: Dict[Tuple[int, float], Tuple[q.qarray, q.qarray, q.qarray]] = {}


# (N, V, hash(geometry)): (time-independent, Omega, Delta)


class EvolvingQubitSystem(BaseQubitSystem):
    def __init__(self, N: int, V: float, geometry: BaseGeometry,
                 Omega: np.ndarray, Delta: np.ndarray,
                 t_list: np.ndarray,
                 ghz_state: BaseGHZState,
                 solve_points_per_timestep: int = 1):
        super().__init__(N, V, geometry)

        self.Omega = Omega
        self.Delta = Delta
        self.t_list = t_list

        assert len(t_list) - 1 == len(Omega) == len(Delta), \
            "Omega and Delta need to be of equal length, and of length one less than t_list"

        self.psi_0 = q.kron(*states_quimb.get_ground_states(N))
        self.ghz_state = ghz_state

        self.solve_points_per_timestep = solve_points_per_timestep

        self.evo: Optional[q.Evolution] = None
        self.solved_states: List[q.qarray] = []
        self.solved_t_list = []

    def get_hamiltonian(self, Omega: float, Delta: float) -> q.qarray:
        key = (self.N, self.V, hash(self.geometry))
        global cached_hamiltonian_variables
        if key in cached_hamiltonian_variables:
            ti, Omega_var, Delta_var = cached_hamiltonian_variables[key]
        else:
            logger.info(f"Calculating Hamiltonian variables for N {self.N}, V {self.V:.2e}")
            hamiltonian_variables = self._get_hamiltonian_variables()
            cached_hamiltonian_variables[key] = hamiltonian_variables
            logger.info(f"Added Hamiltonian variables for N {self.N}, V {self.V:.2e}")
            ti, Omega_var, Delta_var = hamiltonian_variables

        return ti + Omega * Omega_var + Delta * Delta_var

    def _get_hamiltonian_variables(self) -> Tuple[q.qarray, q.qarray, q.qarray]:
        sx = q.pauli("X")
        sz = q.pauli("Z")
        qnum = (sz + q.identity(2)) / 2
        dims = [2] * self.N

        # noinspection PyTypeChecker
        time_independent_terms: q.qarray = 0
        # noinspection PyTypeChecker
        Omega_coeff_terms: q.qarray = 0
        # noinspection PyTypeChecker
        Delta_coeff_terms: q.qarray = 0

        for i in range(self.N):
            Omega_coeff_terms += q.ikron(sx, dims=dims, inds=i) / 2
            n_i = q.ikron(qnum, dims=dims, inds=i)
            Delta_coeff_terms -= n_i

            for j in range(i):
                n_j = q.ikron(qnum, dims=dims, inds=j)

                time_independent_terms += self.V / self.geometry.get_distance(i, j) ** 6 * n_i * n_j

        return (
            time_independent_terms,
            Omega_coeff_terms,
            Delta_coeff_terms
        )

    def solve(self) -> List[q.qarray]:
        dt = self.t_list[1]

        self.solved_states = [self.psi_0]
        self.solved_t_list = [0]

        latest_state = state = self.psi_0
        latest_time = 0
        for i in tqdm(range(len(self.Omega))):
            Omega = self.Omega[i]
            Delta = self.Delta[i]
            self.evo = q.Evolution(
                latest_state,
                self.get_hamiltonian(Omega, Delta),
                # method="expm",
                # progbar=True,
            )
            solve_points = np.linspace(0, dt, self.solve_points_per_timestep + 1)[1:]  # Take away t=0 as a solve point
            for state in self.evo.at_times(solve_points):
                self.solved_states.append(state)
            self.solved_t_list += (solve_points + latest_time).tolist()
            latest_state = state
            latest_time = self.solved_t_list[-1]

        return self.solved_states

    @plotting_decorator
    def plot(self, with_antisymmetric_ghz: bool = False, fig_kwargs: dict = None, plot_titles: bool = True,
             plot_others_as_sum: bool = False):
        assert self.evo is not None, "evo attribute cannot be None (call solve method)"

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

        Omega = interpolate.interp1d(self.t_list, np.hstack((self.Omega, self.Omega[-1])), kind="previous",
                                     fill_value="extrapolate")
        Delta = interpolate.interp1d(self.t_list, np.hstack((self.Delta, self.Delta[-1])), kind="previous",
                                     fill_value="extrapolate")

        ax.plot(self.solved_t_list, [Omega(t) for t in self.solved_t_list], label=r"$\Omega{}$", lw=3, alpha=0.8)
        ax.plot(self.solved_t_list, [Delta(t) for t in self.solved_t_list], label=r"$\Delta{}$", lw=3, alpha=0.8)
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
                self.solved_t_list,
                [q.fidelity(_state, _instantaneous_state)
                 for _instantaneous_state in self.solved_states],
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
        states = states_quimb.get_states(self.N) if not plot_others_as_sum \
            else [states_quimb.get_excited_states(self.N), states_quimb.get_ground_states(self.N)]
        fidelities = []

        plot_individual_orthogonal_state_labels = len(states) <= 4
        plotted_others = False
        for i, state in enumerate(tqdm(states)):
            label = states_quimb.get_label_from_state(state)
            state_product_basis_index = states_quimb.get_product_basis_states_index(state)
            state_fidelities = [np.abs(_instantaneous_state.flatten()[state_product_basis_index]) ** 2
                                for _instantaneous_state in self.solved_states]

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
                self.solved_t_list,
                state_fidelities,
                label=plot_label,
                color='g' if 'e' not in label else 'r' if 'g' not in label else 'k',
                linewidth=1 if 'e' not in label or 'g' not in label else 0.5,
                alpha=0.5
            )

        fidelities_sum = np.array(fidelities).sum(axis=0)
        ax.plot(self.solved_t_list, fidelities_sum,
                label="$P_{E} + P_{G}$",
                color='C0', linestyle=":", linewidth=1, alpha=0.7)

        if plot_others_as_sum:
            others_sum = 1 - fidelities_sum

            ax.plot(self.solved_t_list, others_sum,
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

    def get_fidelity_with(self, target_state: Union[str, q.qarray] = "ghz") -> float:
        """
        :param target_state: One of "ghz", "ghz_antisymmetric", "ground", and "excited"
        :return:
        """
        assert (self.evo is not None), "evo attribute cannot be None (call solve method)"

        final_state = self.solved_states[-1]
        if target_state == "ghz":
            return q.fidelity(final_state, self.ghz_state.get_state_tensor(symmetric=True))
        elif target_state == "ghz_antisymmetric":
            return q.fidelity(final_state, self.ghz_state.get_state_tensor(symmetric=False))
        elif target_state == "ground":
            return q.fidelity(final_state, q.kron(*states_quimb.get_ground_states(self.N)))
        elif target_state == "excited":
            return q.fidelity(final_state, q.kron(*states_quimb.get_excited_states(self.N)))
        elif isinstance(target_state, q.qarray):
            return q.fidelity(final_state, target_state)
        else:
            raise ValueError(f"target_state has to be one of 'ghz', 'ground', or 'excited', not {target_state}.")


if __name__ == '__main__':
    from qubit_system.geometry.regular_lattice_1d import RegularLattice1D

    # s_qs = StaticQubitSystem(
    #     N=4, V=1,
    #     geometry=RegularLattice1D(),
    #     Omega=1, Delta=np.linspace(-1, 1, 50)
    # )
    # s_qs.plot()

    from qubit_system.utils.ghz_states_quimb import StandardGHZState

    t = 1
    N = 20
    e_qs = TimeIndependentEvolvingQubitSystem(
        N=N, V=1, geometry=RegularLattice1D(),
        Omega=1, Delta=1,
        t_list=np.linspace(0, 1, 20),
        ghz_state=StandardGHZState(N)
    )
    e_qs.solve()
    e_qs.plot(with_antisymmetric_ghz=True, show=True)

    # t = 1
    # N = 4
    # t_num = 30
    # e_qs = EvolvingQubitSystem(
    #     N=N, V=1, geometry=RegularLattice1D(),
    #     Omega=np.ones(t_num - 1),
    #     Delta=np.linspace(-1, 1, t_num - 1),
    #     t_list=np.linspace(0, 1, t_num),
    #     ghz_state=StandardGHZState(N),
    #     solve_points_per_timestep=50
    # )
    # e_qs.solve()
    # e_qs.plot(with_antisymmetric_ghz=True, show=True)
