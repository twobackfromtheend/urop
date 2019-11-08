from typing import Tuple, List

import numpy as np
import quimb as q
from matplotlib import pyplot as plt, ticker
from tqdm._tqdm_notebook import tqdm_notebook as tqdm

from qubit_system.geometry import BaseGeometry
from qubit_system.qubit_systems.decorators import plotting_decorator
from qubit_system.qubit_systems.base_qubit_system import BaseQubitSystem
from qubit_system.utils import states


class StaticQubitSystem(BaseQubitSystem):
    def __init__(self, N: int, V: float, geometry: BaseGeometry, Omega: float, Delta: np.ndarray):
        super().__init__(N, V, geometry)

        self.Omega = Omega
        self.Delta = Delta

        self.states = states.get_states(self.N)
        self.Omega_zero_energies = None
        self.Omega_non_zero_energies = None

    def get_hamiltonian(self, detuning: float) -> q.qarray:
        return self._get_hamiltonian(self.Omega, detuning)

    def get_energies(self):
        omega_zero_all_energies = []
        omega_non_zero_all_energies = []

        state_tensors = [q.kron(*state) for state in self.states]
        for detuning in tqdm(self.Delta):
            H = self.get_hamiltonian(detuning)
            energies = []

            for i, state in enumerate(self.states):
                energy = q.expec(H, state_tensors[i]).real
                energies.append(energy)
            omega_zero_all_energies.append(energies)
            if self.Omega != 0:
                eigenvalues = q.eigvalsh(H.toarray()).real
                omega_non_zero_all_energies.append(eigenvalues)
        self.Omega_zero_energies = np.array(omega_zero_all_energies)
        self.Omega_non_zero_energies = np.array(omega_non_zero_all_energies)

    def plot(self):
        self.plot_detuning_energy_levels(
            plot_state_names=self.N <= 4,
            savefig_name=f"detuning_{self.N}.png",
            show=True
        )

    @plotting_decorator
    def plot_detuning_energy_levels(self, plot_state_names: bool, fig_kwargs: dict = None, plot_title: bool = True,
                                    ylim: Tuple[float, float] = None, highlight_states_by_label: List[str] = None):
        if self.Omega_zero_energies is None:
            self.get_energies()

        if highlight_states_by_label is None:
            highlight_states_by_label = ''.join(['e' for _ in range(self.N)])

        if fig_kwargs is None:
            fig_kwargs = {}
        fig_kwargs = {**dict(figsize=(15, 7), num="Energy Levels"), **fig_kwargs}

        plt.figure(**fig_kwargs)

        plot_points = len(self.Delta)
        for i in reversed(range(len(self.states))):
            label = states.get_label_from_state(self.states[i])
            is_highlight_state = label in highlight_states_by_label
            is_ground_state = 'e' not in label
            color = 'g' if is_ground_state else 'r' if is_highlight_state else 'grey'
            linewidth = 5 if is_ground_state or is_highlight_state else 1
            z_order = 2 if is_ground_state or is_highlight_state else 1
            # color = f'C{i}'
            plt.plot(self.Delta, self.Omega_zero_energies[:, i], color=color, label=label, alpha=0.6, lw=linewidth,
                     zorder=z_order)
            if self.Omega != 0:
                plt.plot(self.Delta, self.Omega_non_zero_energies[:, i], color=f'C{i}', ls=':', alpha=0.6)

            if plot_state_names:
                Delta_index = int(plot_points / len(self.states)) * i + int(plot_points / 2 / len(self.states))
                text_x = self.Delta[Delta_index]
                text_y = self.Omega_zero_energies[Delta_index, i]
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