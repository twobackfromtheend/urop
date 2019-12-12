from collections import defaultdict

import quimb as q
from qubit_system.utils import states


def get_f_function_generator(N):
    states_list = states.get_states(N)
    state_tensors_by_excited_count = defaultdict(list)
    for state in states_list:
        state_label = states.get_label_from_state(state)
        state_excited_count = sum(letter == "e" for letter in state_label)
        state_tensors_by_excited_count[state_excited_count].append(q.kron(*state))

    def _get_f_for_excited_count_quimb(count: int):
        figure_of_merit_kets = state_tensors_by_excited_count[count]

        def _get_f_excited_count(state):
            q_state = q.qu(state, sparse=False)
            fidelities = [
                q.fidelity(q_state, fom_ket)
                for fom_ket in figure_of_merit_kets
            ]
            figure_of_merit = sum(fidelities)
            return figure_of_merit

        return _get_f_excited_count

    return _get_f_for_excited_count_quimb
