import os
import pickle
import re
from collections import deque
from datetime import datetime
from operator import attrgetter
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np

from reinforcement_learning.Environments.evolving_qubit_env import EvolvingQubitEnv
from reinforcement_learning.Environments.ti_evolving_qubit_env import TIEvolvingQubitEnv

log_file_env = os.getenv('LOG_FILE')
LOG_FILE = Path(log_file_env) if log_file_env is not None else Path(__file__).parent / "logs" / "s_baselines.log"
RESULTS_FOLDER = Path(__file__).parent / "results"


class Protocol(NamedTuple):
    fidelity: float
    Omega: np.ndarray
    Delta: np.ndarray


def process_log_file(env: Union[EvolvingQubitEnv, TIEvolvingQubitEnv]):
    protocols = []
    with LOG_FILE.open('r') as f:
        previous_lines = deque(maxlen=10)

        for line in f:
            previous_lines.append(line)

            if line.startswith("action"):
                fidelity_achieved_line = previous_lines[-4]
                fidelity_achieved = float(re.search(r"fidelity_achieved: ([\d.]*),", fidelity_achieved_line).group(1))

                _search = re.search(r"'Omega': ([-\d\[\]\., e]*), 'Delta': ([-\d\[\]\., e]*)", line)
                Omega = eval(_search.group(1))
                Delta = eval(_search.group(2))

                Omega = np.array(Omega)
                Delta = np.array(Delta)

                protocols.append(Protocol(fidelity_achieved, Omega, Delta))

    # print(protocols)
    if len(protocols) == 0:
        print("Could not find any protocols.")
        return

    best_protocol = max(protocols, key=attrgetter('fidelity'))

    print(f"Best protocol has fidelity: {best_protocol.fidelity}")

    for _protocol in protocols:
        if _protocol.fidelity == best_protocol and _protocol is not best_protocol:
            print(f"Found another protocol with equal fidelity: {_protocol}")

    data = {'protocol': best_protocol}
    if isinstance(env, EvolvingQubitEnv):
        data['evolving_qubit_system_kwargs'] = env.evolving_qubit_system_kwargs
    elif isinstance(env, TIEvolvingQubitEnv):
        data['ti_evolving_qubit_system_kwargs'] = env.ti_evolving_qubit_system_kwargs
        data['ti_evolving_qubit_system_kwargs']['t_list'] = env.t_list
    else:
        print(f"Unknown env type: {env.__class__.__name__}")

    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_filepath = RESULTS_FOLDER / (current_datetime + ".pkl")

    RESULTS_FOLDER.mkdir(exist_ok=True)

    with data_filepath.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved best protocol and data to {data_filepath.name}")


if __name__ == '__main__':
    from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
    from qubit_system.utils.ghz_states import StandardGHZState

    N = 4
    t = 5
    env = EvolvingQubitEnv(N=N, V=1, geometry=RegularLattice1D(), t_list=np.linspace(0, t, 20),
                           Omega_range=(0, 1), Delta_range=(-1, 1),
                           ghz_state=StandardGHZState(N))

    process_log_file(env)
