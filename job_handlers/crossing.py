from job_handlers.hamiltonian import SpinHamiltonian
from qubit_system.geometry import BaseGeometry
from qubit_system.utils.ghz_states import BaseGHZState

import numpy as np
import quimb as q


def get_ghz_crossing(spin_ham: SpinHamiltonian, characteristic_V: float, ghz_state: BaseGHZState,
                     geometry: BaseGeometry,
                     V: float):
    ghz_1 = ghz_state._get_components()[1]
    Delta_range = np.linspace(-characteristic_V * 5, characteristic_V * 5, 50)

    energies = []
    for detuning in Delta_range:
        H = spin_ham.get_hamiltonian(V, geometry, Omega=0, Delta=detuning)
        energy = q.expec(H, ghz_1).real
        energies.append(energy)
    energies = np.array(energies)

    def find_root(x: np.ndarray, y: np.ndarray):
        """
        Finds crossing (where y equals 0), given that x, y is roughly linear.
        """
        if (y == 0).all():
            return np.nan
        _right_bound = (y < 0).argmax()
        _left_bound = _right_bound - 1
        crossing = y[_left_bound] / (y[_left_bound] - y[_right_bound]) \
                   * (x[_right_bound] - x[_left_bound]) \
                   + x[_left_bound]
        return crossing

    crossing = find_root(Delta_range, energies)
    print(f"Found crossing: {crossing:.3e}")

    return crossing


def get_crossing(state: q.qarray, spin_ham: SpinHamiltonian, characteristic_V: float, geometry: BaseGeometry,
                 V: float):
    Delta_range = np.linspace(-characteristic_V * 5, characteristic_V * 5, 50)

    energies = []
    for detuning in Delta_range:
        H = spin_ham.get_hamiltonian(V, geometry, Omega=0, Delta=detuning)
        energy = q.expec(H, state).real
        energies.append(energy)
    energies = np.array(energies)

    def find_root(x: np.ndarray, y: np.ndarray):
        """
        Finds crossing (where y equals 0), given that x, y is roughly linear.
        """
        if (y == 0).all():
            return np.nan
        _right_bound = (y < 0).argmax()
        _left_bound = _right_bound - 1
        crossing = y[_left_bound] / (y[_left_bound] - y[_right_bound]) \
                   * (x[_right_bound] - x[_left_bound]) \
                   + x[_left_bound]
        return crossing

    crossing = find_root(Delta_range, energies)
    # print(f"Found crossing: {crossing:.3e}")

    return crossing
