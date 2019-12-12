from typing import Dict, Tuple, Optional, List, Union

import numpy as np
import quimb as q
from matplotlib import pyplot as plt, ticker
from scipy import interpolate

from qubit_system.geometry import BaseGeometry
from qubit_system.qubit_system_classes import logger
from qubit_system.qubit_systems.decorators import plotting_decorator
from qubit_system.qubit_systems.base_qubit_system import BaseQubitSystem
from qubit_system.utils import states
from qubit_system.utils.ghz_states import BaseGHZState

cached_hamiltonian_variables: Dict[Tuple[int, float, int], Tuple[q.qarray, q.qarray, q.qarray]] = {}


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

        end_points = 0
        self.Omega = np.concatenate([self.Omega, np.zeros(end_points)])
        # self.Delta = np.concatenate([self.Delta, np.linspace(self.Delta[-1], 0, end_points)])
        self.Delta = np.concatenate([self.Delta, np.ones(end_points) * self.Delta[-1]])
        dt = self.t_list[1]
        self.t_list = np.concatenate([self.t_list, np.arange(1, end_points + 1) * dt + self.t_list[-1]])

        assert len(t_list) - 1 == len(Omega) == len(Delta), \
            "Omega and Delta need to be of equal length, and of length one less than t_list"

        self.psi_0 = q.kron(*states.get_ground_states(N))
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

    def solve(self) -> List[q.qarray]:
        dt = self.t_list[1]

        self.solved_states = [self.psi_0]
        self.solved_t_list = [0]

        latest_state = state = self.psi_0
        latest_time = 0
        # for i in tqdm(range(len(self.Omega))):
        for i in range(len(self.Omega)):
            Omega = self.Omega[i]
            Delta = self.Delta[i]
            self.evo = q.Evolution(
                latest_state,
                self.get_hamiltonian(Omega, Delta),
                # method="expm",
                method="integrate",
                # progbar=True,
            )
            solve_points = np.linspace(0, dt, self.solve_points_per_timestep + 1)[1:]  # Take away t=0 as a solve point
            for state in self.evo.at_times(solve_points):
                self.solved_states.append(state)
            self.solved_t_list += (solve_points + latest_time).tolist()
            latest_state = state
            latest_time = self.solved_t_list[-1]

        # Add some states for end-of-protocol.
        # for i in range(10):
        #     self.evo = q.Evolution(
        #         latest_state,
        #         self.get_hamiltonian(Omega=0, Delta=self.Delta[-1]),
        #         # method="expm",
        #         method="integrate",
        #         # progbar=True,
        #     )
        #     solve_points = np.linspace(0, dt, self.solve_points_per_timestep + 1)[1:]  # Take away t=0 as a solve point
        #     for state in self.evo.at_times(solve_points):
        #         self.solved_states.append(state)
        #     self.solved_t_list += (solve_points + latest_time).tolist()
        #     latest_state = state
        #     latest_time = self.solved_t_list[-1]

        return self.solved_states

    @plotting_decorator
    def plot(self, with_antisymmetric_ghz: bool = False, fig_kwargs: dict = None, plot_titles: bool = True,
             plot_others_as_sum: bool = False):
        assert self.evo is not None, "evo attribute cannot be None (call solve method)"

        if fig_kwargs is None:
            fig_kwargs = {}
        fig_kwargs = {**dict(figsize=(12, 9)), **fig_kwargs}

        fig, axs = plt.subplots(4, 1, sharex='all', **fig_kwargs)
        self.plot_Omega_and_Delta(axs[0], plot_title=plot_titles)
        self.plot_ghz_states_overlaps(axs[1], with_antisymmetric_ghz, plot_title=plot_titles)
        self.plot_basis_states_overlaps(axs[2], plot_title=plot_titles, plot_others_as_sum=plot_others_as_sum)
        self.plot_entanglement_entropies(axs[3], plot_title=plot_titles)

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
        if with_antisymmetric_ghz:
            labelled_states.append(
                (self.ghz_state.get_state_tensor(symmetric=False), r"$\psi_{\mathrm{GHZ}}^{\mathrm{a}}$"))

        for _state, _label in labelled_states:
            ax.plot(
                self.solved_t_list,
                [q.fidelity(_state, _instantaneous_state)
                 for _instantaneous_state in self.solved_states],
                # label=_label,
                lw=1,
                alpha=0.8
            )
        ax.set_ylabel(r"GHZ Fidelity ${\left| \langle \mathrm{GHZ} | \psi_k (t) \rangle \right| }^2$")
        if plot_title:
            ax.set_title("Fidelity with GHZ states")
        ax.set_ylim((-0.1, 1.1))
        ax.yaxis.set_ticks([0, 0.5, 1])
        # ax.legend()

    def plot_basis_states_overlaps(self, ax, plot_title: bool = True, plot_others_as_sum: bool = False):
        states_list = states.get_states(self.N) if not plot_others_as_sum \
            else [states.get_excited_states(self.N), states.get_ground_states(self.N)]
        fidelities = []

        plot_individual_orthogonal_state_labels = len(states_list) <= 4
        plotted_others = False
        # for i, state in enumerate(tqdm(states)):
        for i, state in enumerate(states_list):
            label = states.get_label_from_state(state)
            state_product_basis_index = states.get_product_basis_states_index(state)
            state_fidelities = np.array([np.abs(_instantaneous_state.flatten()[state_product_basis_index]) ** 2
                                for _instantaneous_state in self.solved_states])

            if 'e' not in label or 'g' not in label:
                fidelities.append(state_fidelities)

            if ('e' not in label or 'g' not in label):
                plot_label = r"$P_{" + f"{label.upper()[0]}" + "}$"
            elif plot_individual_orthogonal_state_labels:
                plot_label = r"$P_{" + f"{label.upper()}" + "}$"
            else:
                plot_label = 'Others'

            if plot_label == 'Others':
                if plotted_others:
                    plot_label = None
                else:
                    plotted_others = True

            if (state_fidelities> 0.45).any():
                print(label, max(state_fidelities))
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

        ax.set_ylabel("Population")
        if plot_title:
            ax.set_title("Basis state populations")
        ax.set_ylim((-0.1, 1.1))
        ax.yaxis.set_ticks([0, 0.5, 1])

        # ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles)))
        ax.legend(handles, labels)

    def plot_entanglement_entropies(self, ax, plot_title: bool = True):
        subsystem_entropy = [q.entropy_subsys(state_, [2] * self.N, np.arange(self.N / 2))
                             for state_ in self.solved_states]
        label = r"$\mathcal{S}\, ( \rho_A )$"
        # ax.plot(self.solved_t_list, subsystem_entropy, label=label)
        ax.plot(self.solved_t_list, subsystem_entropy)

        ax.set_ylabel(label)
        if plot_title:
            ax.set_title("Entanglement Entropy")
        ax.set_ylim((0, self.N / 2))
        # ax.legend()

    def get_fidelity_with(self, target_state: Union[str, q.qarray] = "ghz") -> float:
        """
        :param target_state:
            One of "ghz", "ghz_antisymmetric", "ground", and "excited".
            Can also be ghz_component_1 or ghz_component_2
        :return:
        """
        assert (self.evo is not None), "evo attribute cannot be None (call solve method)"

        final_state = self.solved_states[-1]
        if target_state == "ghz":
            return q.fidelity(final_state, self.ghz_state.get_state_tensor(symmetric=True))
        elif target_state == "ghz_antisymmetric":
            return q.fidelity(final_state, self.ghz_state.get_state_tensor(symmetric=False))
        elif target_state == "ghz_component_1":
            return q.fidelity(final_state, self.ghz_state._get_components()[0])
        elif target_state == "ghz_component_2":
            return q.fidelity(final_state, self.ghz_state._get_components()[1])
        elif target_state == "ground":
            return q.fidelity(final_state, q.kron(*states.get_ground_states(self.N)))
        elif target_state == "excited":
            return q.fidelity(final_state, q.kron(*states.get_excited_states(self.N)))
        elif isinstance(target_state, q.qarray):
            return q.fidelity(final_state, target_state)
        else:
            raise ValueError(f"target_state has to be one of 'ghz', 'ground', or 'excited', not {target_state}.")
