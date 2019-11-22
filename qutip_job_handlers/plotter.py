from typing import Callable

import matplotlib.pyplot as plt
from matplotlib import ticker
from qutip import *


def plot_ghz_states_overlaps(ax, solve_result, ghz_state, t_list, with_antisymmetric_ghz: bool,
                             plot_title: bool = True):
    labelled_states = [(ghz_state.get_state_tensor(), r"$\psi_{\mathrm{GHZ}}^{\mathrm{s}}$")]
    if with_antisymmetric_ghz is not None:
        labelled_states.append(
            (ghz_state.get_state_tensor(symmetric=False), r"$\psi_{\mathrm{GHZ}}^{\mathrm{a}}$"))

    for _state, _label in labelled_states:
        ax.plot(
            t_list,
            [fidelity(_state, _instantaneous_state) ** 2
             for _instantaneous_state in solve_result.states],
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


def plot_Omega_and_Delta(ax, Omega, Delta, t_list, plot_title: bool = True):
    """
    Plots Omega and Delta as a function of time.
    Includes overlap with GHZ state if `self.solve_result` is not None (self.solve has been called).
    :return:
    """
    ax.xaxis.set_major_formatter(ticker.EngFormatter('s'))
    plt.xlabel('Time')

    ax.plot(t_list, [Omega(t) for t in t_list], label=r"$\Omega{}$", lw=3, alpha=0.8)
    ax.plot(t_list, [Delta(t) for t in t_list], label=r"$\Delta{}$", lw=3, alpha=0.8)
    ax.yaxis.set_major_formatter(ticker.EngFormatter('Hz'))
    ax.locator_params(nbins=4, axis='y')
    ax.locator_params(nbins=5, axis='x')

    ax.legend()
    if plot_title:
        ax.set_title("Control parameters")
    delta = (t_list.max() - t_list.min()) * 0.01
    ax.set_xlim((t_list.min() - delta, t_list.max() + delta))


# def plot_statistics(ax, solve_result):
#     # ax.plot(t_list, [participation_ratio(state) for state in solve_result.states], label="Participation Ratio")
#     N = len((solve_result.states[0]).dims[0])
#     half_N = int(N / 2)
#
#     def entropy_subsys(state: Qobj):
#         ptr = state.ptrace(np.arange(half_N))
#         return entropy_vn(ptr, base=2)
#
#     ax.plot(t_list, [entropy_subsys(state) for state in solve_result.states], label="Entanglement Entropy")
#     ax.legend()
#     ax.set_ylim((0, half_N))
#     ax.locator_params(nbins=4, axis='y')
#     ax.set_title("Statistics")


def plot(solve_result, ghz_state, t_list, Omega: Callable, Delta: Callable, with_antisymmetric_ghz: bool = False, fig_kwargs: dict = None,
         plot_titles: bool = True, ):
    if fig_kwargs is None:
        fig_kwargs = {}
    fig_kwargs = {**dict(figsize=(15, 9)), **fig_kwargs}

    fig, axs = plt.subplots(3, 1, sharex='all', **fig_kwargs)
    plot_Omega_and_Delta(axs[0], Omega, Delta, t_list, plot_title=plot_titles)
    plot_ghz_states_overlaps(axs[1], solve_result, ghz_state, t_list, with_antisymmetric_ghz, plot_title=plot_titles)
    # plot_basis_states_overlaps(axs[2], plot_title=plot_titles, plot_others_as_sum=plot_others_as_sum)
    # plot_statistics(axs[2], solve_result)
    plt.xlabel('Time')
    plt.tight_layout()

    for ax in axs:
        ax.grid()

        # Move legend to the right
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
