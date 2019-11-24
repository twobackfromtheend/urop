import os
import numpy as np
from qutip import Options, Qobj, sesolve

options_kwargs = {
    "num_cpus": os.getenv("NCPUS", 0),
    "store_final_state": True,
    "nsteps": 5000,
    "use_openmp": True
}

print(f"options kwargs: {options_kwargs}")

qutip_options = Options(**options_kwargs)

print(f"options: {str(qutip_options)}")


def solve(hamiltonian: list, psi_0: Qobj, t_list: np.ndarray):
    # noinspection PyTypeChecker
    return sesolve(
        hamiltonian,
        psi_0,
        t_list,
        options=Options(store_states=True, nsteps=5000),
    )
