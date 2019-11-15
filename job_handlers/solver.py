from typing import List

import quimb as q
import numpy as np

from job_handlers.hamiltonian import QType, SpinHamiltonian
from qubit_system.geometry import BaseGeometry


def solve_with_protocol(spin_ham: SpinHamiltonian,
                        V: float, geometry: BaseGeometry,
                        t_list: np.ndarray,
                        psi_0: QType,
                        Omega: np.ndarray, Delta: np.ndarray) -> List[q.qarray]:
    dt = t_list[1]
    latest_state = psi_0
    states = [latest_state]
    for i in range(len(Omega)):
        _Omega = Omega[i]
        _Delta = Delta[i]
        evo = q.Evolution(
            latest_state,
            spin_ham.get_hamiltonian(V, geometry, _Omega, _Delta),
            method="integrate",
        )
        evo.update_to(dt)
        latest_state = evo.pt
        states.append(latest_state)

    return states
