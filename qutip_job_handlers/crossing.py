import numpy as np
from qutip import expect

from qubit_system.geometry import BaseGeometry
from qutip_job_handlers.hamiltonian import QutipSpinHamiltonian
from qutip_job_handlers.qutip_states import BaseGHZState


def get_ghz_crossing(spin_ham: QutipSpinHamiltonian, characteristic_V: float, ghz_state: BaseGHZState,
                     geometry: BaseGeometry,
                     V: float):
    ghz_1 = ghz_state._get_components()[1]
    Delta_range = np.linspace(-characteristic_V * 5, characteristic_V * 5, 50)

    energies = []
    H_list = spin_ham.get_hamiltonian(V, geometry, Omega=0, Delta=0)
    for detuning in Delta_range:
        H = H_list[0] + H_list[2][0] * detuning
        energy = expect(H, ghz_1).real
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
