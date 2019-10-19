from typing import Optional, List, Union

import numpy as np
import quimb as q
from matplotlib import pyplot as plt, ticker
from tqdm._tqdm_notebook import tqdm_notebook as tqdm

from qubit_system.geometry import BaseGeometry
from qubit_system.qubit_systems.decorators import plotting_decorator
from qubit_system.qubit_systems.base_qubit_system import BaseQubitSystem
from qubit_system.utils import states
from qubit_system.utils.ghz_states import BaseGHZState


class TimeIndependentEvolvingQubitSystem(BaseQubitSystem):
    def __init__(self, N: int, V: float, geometry: BaseGeometry,
                 Omega: float, Delta: float,
                 t_list: np.ndarray, ghz_state: BaseGHZState, psi_0: q.qarray = None):
        super().__init__(N, V, geometry)

        self.Omega = Omega
        self.Delta = Delta
        self.t_list = t_list

        self.psi_0 = q.kron(*states.get_ground_states(N, sparse=True)) if psi_0 is None else psi_0
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
        states = states.get_states(self.N) if not plot_others_as_sum \
            else [states.get_excited_states(self.N), states.get_ground_states(self.N)]
        fidelities = []

        plot_individual_orthogonal_state_labels = len(states) <= 4
        plotted_others = False
        for i, state in enumerate(tqdm(states)):
            label = states.get_label_from_state(state)
            state_product_basis_index = states.get_product_basis_states_index(state)
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
            return q.fidelity(final_state, q.kron(*states.get_ground_states(self.N)))
        elif target_state == "excited":
            return q.fidelity(final_state, q.kron(*states.get_excited_states(self.N)))
        elif isinstance(target_state, q.qarray):
            return q.fidelity(final_state, target_state)
        else:
            raise ValueError(f"target_state has to be one of 'ghz', 'ground', or 'excited', not {target_state}.")
