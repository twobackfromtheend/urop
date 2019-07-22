import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib import ticker
from qutip import fidelity, tensor, expect, Qobj
from qutip.solver import Result

from hamiltonian import get_hamiltonian
from states import get_states, get_label_from_state

PLOT_FOLDER = 'plots/'


def plotting_decorator(func):
    def wrapper(*args, show: bool = False, savefig_name: str = None, **kwargs):
        return_value = func(*args, **kwargs)

        if savefig_name:
            plt.savefig(PLOT_FOLDER + savefig_name, dpi=300)
        if show:
            plt.show()

        return return_value

    return wrapper


@plotting_decorator
def plot_Omega_and_Delta(t_list, Omega, Delta, ghz_state: Qobj = None, result: Result = None):
    """
    Plots Omega and Delta as a function of time.
    Includes overlap with GHZ state if both `ghz_state` and `result` are passed.
    :param t_list:
    :param Omega:
    :param Delta:
    :param ghz_state:
    :param result:
    :return:
    """
    assert (
            (ghz_state is None and result is None) or
            (ghz_state is not None and result is not None)
    ), "Either both or neither of `ghz_state` and `result` must be passed."

    rows = 2 if ghz_state is None else 3
    fig, axs = plt.subplots(rows, 1, sharex='all', figsize=(15, rows * 3), num="Omega and Delta")
    plt.xlabel('Time')

    ax0 = axs[0]
    ax0.plot(t_list, [Omega(t) for t in t_list])
    ax0.set_ylabel("Omega")
    ax0.yaxis.set_major_formatter(ticker.EngFormatter('Hz'))

    ax1 = axs[1]
    ax1.plot(t_list, [Delta(t) for t in t_list])
    ax1.set_ylabel("Delta")
    ax1.yaxis.set_major_formatter(ticker.EngFormatter('Hz'))

    ax0.xaxis.set_major_formatter(ticker.EngFormatter('s'))

    if ghz_state is not None:
        ax2 = axs[2]
        ax2.plot(t_list, [fidelity(ghz_state, _instantaneous_state) for _instantaneous_state in result.states])
        ax2.set_ylabel("Fidelity with GHZ")
        ax2.set_ylim((-0.1, 1.1))
        ax2.yaxis.set_ticks([0, 1])

    for ax in axs:
        ax.grid()

    plt.tight_layout()


@plotting_decorator
def plot_state_overlaps(L: int, t_list: np.ndarray, result: Result):
    states = get_states(L)
    fig, axs = plt.subplots(len(states), 1, sharex='all', figsize=(15, 3 * 2 ** L + 3), num="State overlaps")

    for i, state in enumerate(states):
        ax = axs[i]
        ax.plot(t_list, [fidelity(tensor(state), _instantaneous_state) for _instantaneous_state in result.states])
        label = get_label_from_state(state)
        ax.set_ylabel(label)
        ax.set_ylim((-0.1, 1.1))
        ax.yaxis.set_ticks([0, 1])
        ax.grid()

    axs[0].xaxis.set_major_formatter(ticker.EngFormatter('s'))
    plt.tight_layout()


@plotting_decorator
def plot_detuning_energy_levels(L: int, V: float, detuning: np.ndarray, plot_state_names: bool):
    states = get_states(L)

    plot_points = len(detuning)

    # Energies for Omega = 0
    all_energies = []
    for i, Delta in tqdm.tqdm(enumerate(detuning)):
        H = get_hamiltonian(L, 0, Delta, V)
        energies = []
        for state in states:
            energy = expect(H, tensor(*state))
            energies.append(energy)
        all_energies.append(energies)
        # eigenvalues, eigenstates = H.eigenstates()
        # all_energies.append(eigenvalues)
    all_energies = np.array(all_energies)

    plt.figure(figsize=(15, 7), num="Energy Levels")

    # plt.plot(detuning, all_energies, 'b', alpha=0.6)
    for i in reversed(range(len(states))):
        label = get_label_from_state(states[i])
        # color = f'C{i}'
        color = 'g' if 'e' not in label else 'r' if 'g' not in label else 'grey'
        plt.plot(detuning, all_energies[:, i], color=color, label=label, alpha=0.6)
        # plt.plot(detuning, non_zero_omega_energies[:, i], color=f'C{i}', ls=':', alpha=0.6)
        if plot_state_names:
            detuning_index = int(plot_points / len(states)) * i + int(plot_points / 2 / len(states))
            text_x = detuning[detuning_index]
            text_y = all_energies[detuning_index, i]
            plt.text(text_x, text_y, label, ha='center', color=f'C{i}', fontsize=16, fontweight='bold')

    if plot_state_names:
        plt.legend()

    plt.grid()

    plt.title(f"Plot of eigenvalues of $H$ with $V = {V:0.3e}$ ")
    plt.xlabel("Detuning $\Delta$ (Hz)")
    plt.ylabel("Energy (Hz)")
    plt.tight_layout()
