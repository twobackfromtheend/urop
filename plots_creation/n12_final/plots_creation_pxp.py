import random
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import quimb as q

from job_handlers.timer import timer
from plots_creation.n12_final.utils import save_current_fig
from qubit_system.utils.ghz_states import CustomGHZState, StandardGHZState, AlternatingGHZState


def _create_pxp_hamiltonian_1d(N: int):
    ket_0 = q.ket([0, 1])
    ket_1 = q.ket([1, 0])
    P = ket_0 @ ket_0.H
    # X = ket_0 @ ket_1.H + ket_1 @ ket_0.H
    X = q.pauli('x')

    with timer("Generating PXP Hamiltonian"):
        dims = [2] * N
        pxp_hamiltonian = 0
        for i in range(N):
            # Open boundary conditions
            if i == 0:
                X_i1 = q.ikron(X, dims=dims, inds=i, sparse=True)
                P_i2 = q.ikron(P, dims=dims, inds=i + 1, sparse=True)
                pxp_hamiltonian += X_i1 @ P_i2
                continue
            elif i == N - 1:
                P_i0 = q.ikron(P, dims=dims, inds=i - 1, sparse=True)
                X_i1 = q.ikron(X, dims=dims, inds=i, sparse=True)
                pxp_hamiltonian += P_i0 @ X_i1
                continue
            P_i0 = q.ikron(P, dims=dims, inds=i - 1, sparse=True)
            X_i1 = q.ikron(X, dims=dims, inds=i, sparse=True)
            P_i2 = q.ikron(P, dims=dims, inds=i + 1, sparse=True)
            pxp_hamiltonian += P_i0 @ X_i1 @ P_i2

    return pxp_hamiltonian


def _create_pxp_hamiltonian_2d(shape: Tuple[int, int]):
    N = 1
    for _shape in shape:
        assert isinstance(_shape, int)
        N *= _shape

    ket_0 = q.ket([0, 1])
    ket_1 = q.ket([1, 0])
    P = ket_0 @ ket_0.H
    X = q.pauli('x')

    cols = shape[0]
    rows = shape[1]

    with timer("Generating PXP Hamiltonian"):
        dims = [2] * N
        pxp_hamiltonian = 0
        for i in range(N):
            col = i // rows
            row = i - rows * col
            print(i, row, col)

            P_i0 = P_i2 = P_j0 = P_j2 = q.identity(d=2 ** N, sparse=True)

            if row == 0:
                P_i2 = q.ikron(P, dims=dims, sparse=True,
                               inds=i + 1)
            elif row == rows - 1:
                P_i0 = q.ikron(P, dims=dims, sparse=True,
                               inds=i - 1)
            else:
                P_i2 = q.ikron(P, dims=dims, sparse=True,
                               inds=i + 1)
                P_i0 = q.ikron(P, dims=dims, sparse=True,
                               inds=i - 1)
            if col == 0:
                P_j2 = q.ikron(P, dims=dims, sparse=True,
                               inds=(col + 1) * rows + row)
            elif col == cols - 1:
                P_j0 = q.ikron(P, dims=dims, sparse=True,
                               inds=(col - 1) * rows + row)
            else:
                P_j2 = q.ikron(P, dims=dims, sparse=True,
                               inds=(col + 1) * rows + row)
                P_j0 = q.ikron(P, dims=dims, sparse=True,
                               inds=(col - 1) * rows + row)

            X_i1 = q.ikron(X, dims=dims, inds=i, sparse=True)
            pxp_hamiltonian += P_i0 @ P_j0 @ X_i1 @ P_j2 @ P_i2

    return pxp_hamiltonian


def _create_pxp_hamiltonian_3d(shape: Tuple[int, int, int]):
    N = 1
    for _shape in shape:
        assert isinstance(_shape, int)
        N *= _shape

    ket_0 = q.ket([0, 1])
    ket_1 = q.ket([1, 0])
    P = ket_0 @ ket_0.H
    X = q.pauli('x')

    x_len = shape[0]
    y_len = shape[1]
    z_len = shape[2]

    with timer("Generating PXP Hamiltonian"):
        dims = [2] * N
        pxp_hamiltonian = 0
        for i in range(N):
            x_i = i // (y_len * z_len)
            y_i = (i - x_i * (y_len * z_len)) // (z_len)
            z_i = (i - x_i * (y_len * z_len) - y_i * z_len)

            P_i0 = P_i2 = P_j0 = P_j2 = P_k0 = P_k2 = q.identity(d=2 ** N)

            if x_i == 0:
                P_i2 = q.ikron(P, dims=dims, sparse=True,
                               inds=i + y_len * z_len)
            elif x_i == x_len - 1:
                P_i0 = q.ikron(P, dims=dims, sparse=True,
                               inds=i - y_len * z_len)
            else:
                P_i2 = q.ikron(P, dims=dims, sparse=True,
                               inds=i + y_len * z_len)
                P_i0 = q.ikron(P, dims=dims, sparse=True,
                               inds=i - y_len * z_len)
            if y_i == 0:
                P_j2 = q.ikron(P, dims=dims, sparse=True,
                               inds=i + z_len)
            elif y_i == y_len - 1:
                P_j0 = q.ikron(P, dims=dims, sparse=True,
                               inds=i - z_len)
            else:
                P_j2 = q.ikron(P, dims=dims, sparse=True,
                               inds=i + z_len)
                P_j0 = q.ikron(P, dims=dims, sparse=True,
                               inds=i - z_len)
            if z_i == 0:
                P_k2 = q.ikron(P, dims=dims, sparse=True,
                               inds=i + 1)
            elif z_i == z_len - 1:
                P_k0 = q.ikron(P, dims=dims, sparse=True,
                               inds=i - 1)
            else:
                P_k2 = q.ikron(P, dims=dims, sparse=True,
                               inds=i + 1)
                P_k0 = q.ikron(P, dims=dims, sparse=True,
                               inds=i - 1)

            X_i1 = q.ikron(X, dims=dims, inds=i, sparse=True)
            pxp_hamiltonian += P_i0 @ P_j0 @ P_k0 @ X_i1 @ P_k2 @ P_j2 @ P_i2

    return pxp_hamiltonian


def create_pxp_plots():
    N = 12
    # pxp_hamiltonian = _create_pxp_hamiltonian_1d(N)
    # pxp_hamiltonian = _create_pxp_hamiltonian_2d((4, 3))
    pxp_hamiltonian = _create_pxp_hamiltonian_3d((3, 2, 2))
    pxp_eigenenergies, pxp_eigenstates = q.eigh(q.qu(pxp_hamiltonian, sparse=False))
    # pxp_eigenenergies, pxp_eigenstates = q.eigh(q.qu(pxp_hamiltonian, sparse=False), autoblock=True)

    pxp_eigenstates = pxp_eigenstates.T

    # 2d
    # ghz_state_alt = CustomGHZState(N, single_component=[True, False, True, False, True, False, True, False, True, False, True, False])
    # ghz_state_std = CustomGHZState(N, single_component=[True, True, True, True, True, True, True, True, True, True, True, True])

    # 3d
    ghz_state_alt = CustomGHZState(N, single_component=[True, False, False, True, False, True, True, False, True, False,
                                                        False, True])
    ghz_state_std = CustomGHZState(N,
                                   single_component=[True, True, True, True, True, True, True, True, True, True, True,
                                                     True])

    z2, z2_prime = ghz_state_alt._get_components()

    def get_random_state(N: int):
        random_bits = random.getrandbits(N)
        bit_string = f"{random_bits:b}".zfill(N)
        ghz_state = CustomGHZState(N, [letter == "1" for letter in bit_string])
        latex_bit_string = bit_string.replace("1", r"◒").replace("0", r"◓")
        state_name = r"$|" + latex_bit_string + r"\rangle$"
        print(state_name)
        return state_name, ghz_state.get_state_tensor()

    random_ghz_states = [get_random_state(N) for _ in range(2)]

    all_states = [
                     (r'$\ket{\mathbb{Z}_2} \equiv \ket{A_N}$', z2),
                     (r'$\ghzalt$', ghz_state_alt.get_state_tensor()),
                     (r'$\ghzstd$', ghz_state_std.get_state_tensor()),
                 ] + random_ghz_states

    states_count = len(all_states)

    fig, axs = plt.subplots(
        2, states_count,
        gridspec_kw={
            'hspace': 0.01, 'height_ratios': [10, 1],
            'top': 0.9, 'bottom': 0.13, 'left': 0.09, 'right': 0.97,
        },
        sharex='col',
        figsize=(12, 4)
    )

    scatter_kwargs = {
        'alpha': 0.3, 's': 12, 'edgecolors': 'none'
    }
    for i, (name, state) in enumerate(all_states):
        state = q.qu(state, sparse=False)
        eigenstate_overlaps = []
        for eigenenergy, eigenvector in zip(pxp_eigenenergies, pxp_eigenstates):
            eigenstate_overlap = q.fidelity(state, eigenvector)
            eigenstate_overlaps.append(eigenstate_overlap)
        eigenstate_overlaps = np.array(eigenstate_overlaps)

        ax = axs[0, i]
        ax.set_yscale('log', basey=10)
        y_lim = 1e-8
        ax.set_ylim((y_lim, 2))

        ax.scatter(pxp_eigenenergies, eigenstate_overlaps, **scatter_kwargs)

        ax1 = axs[1, i]
        masked_i = eigenstate_overlaps < y_lim
        ax1.scatter(pxp_eigenenergies[masked_i], eigenstate_overlaps[masked_i], **scatter_kwargs)
        ax1.set_ylim((0, y_lim))
        ax1.set_yticks([0])

        if i < 3:
            print("a", name)
            ax.set_title(name)
        else:
            print("b", name)
            ax.set_title(name, usetex=False)

        if i != 0:
            ax.get_yaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)

    xlabel = "$E$"
    ylabel = r"${\left| \braket{\psi_{\mathrm{title}} | \psi_k } \right| }^2$"
    fig.text(0.5, 0.04, xlabel, ha='center', va='center')
    fig.text(0.04, 0.5, ylabel, ha='center', va='center', rotation='vertical')
    save_current_fig(f"pxp_crown")


if __name__ == '__main__':
    create_pxp_plots()
