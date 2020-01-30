import datetime
import json
from pathlib import Path
from typing import List

import numpy as np

from qubit_system.geometry import *
from qubit_system.qubit_systems.evolving_qubit_system import EvolvingQubitSystem
from qubit_system.utils.ghz_states import CustomGHZState

TIMESTEPS = 3000

SAVE_FOLDER = Path(__file__).parent


def save_protocol(name: str,
                  N: int, V: float,
                  geometry: BaseGeometry, GHZ_single_component: List[bool],
                  Omega: np.ndarray, Delta: np.ndarray, t_list: np.ndarray,
                  fidelity: float
                  ):
    assert len(Omega) == len(Delta) == len(t_list) - 1 == TIMESTEPS, \
        f"Omega and Delta need to be of equal length, and of length one less than t_list, all equal to {3000}"

    data = {
        'name': name,
        'N': N,
        'V': V,
        'geometry_class': geometry.__class__.__name__,
        'geometry_spacing': geometry.spacing,
        'GHZ_single_component': GHZ_single_component,
        'Omega': Omega.tolist(),
        'Delta': Delta.tolist(),
        't_list': t_list.tolist(),
        'fidelity': fidelity
    }
    if hasattr(geometry, 'shape'):
        data['geometry_shape'] = geometry.shape

    timestamp = datetime.datetime.now().isoformat().replace(":", "-").replace("/", "-")
    save_filename = f"{name}_{fidelity:.3f}_{timestamp}.json"
    with open(SAVE_FOLDER / save_filename, 'w') as f:
        json.dump(data, f)

    print(f"Saved {save_filename}")


def load(filename: str, solve: bool = True):
    try:
        with open(SAVE_FOLDER / filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        file = list(SAVE_FOLDER.glob(filename + "*"))[0]
        with file.open("r") as f:
            data = json.load(f)

    N = data['N']
    V = data['V']
    spacing = data['geometry_spacing']
    shape = data.get('geometry_shape', None)
    if shape is not None:
        geometry = eval(data['geometry_class'])(shape, spacing)
    else:
        geometry = eval(data['geometry_class'])(N, spacing)
    ghz_single_component = data['GHZ_single_component']

    e_qs = EvolvingQubitSystem(
        N=N, V=V, geometry=geometry,
        Omega=data['Omega'],
        Delta=data['Delta'],
        t_list=data['t_list'],
        ghz_state=CustomGHZState(N, ghz_single_component),
    )
    if solve:
        e_qs.solve()
        print(f"Loaded e_qs {filename} with fidelity {e_qs.get_fidelity_with('ghz'):.3f}")
    else:
        print(f"Loaded e_qs {filename} (unsolved).")
    return e_qs
