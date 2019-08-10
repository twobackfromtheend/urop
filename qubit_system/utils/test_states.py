import unittest

from qutip import *

from .states import get_states, get_ground_states, get_excited_states, is_excited


class MyTestCase(unittest.TestCase):
    def test_state_generation_self_consistent(self):
        N = 1
        g = get_ground_states(N)[0]
        e = get_excited_states(N)[0]

        self.assertFalse(is_excited(g))
        self.assertTrue(is_excited(e))

    def test_get_states(self):
        # Test that N = 2 gives "gg" "eg" "ge" "ee" (order not guaranteed)
        N = 2
        states_N_2 = [tensor(_state) for _state in get_states(N)]
        for _state in [ket("gg"), ket("eg"), ket("ge"), ket("ee")]:
            found_state = False
            for _generated_state in states_N_2:
                if _generated_state == _state:
                    found_state = True
                    break
            self.assertTrue(found_state)

        for N in [2, 4, 8]:
            generated_states = get_states(N)
            self.assertEqual(len(generated_states), 2 ** N)  # Ensure correct number of states generated

            # Ensure all-ground and all-excited are generated
            for state in [ket("".join(["g" for _ in range(N)])), ket("".join(["e" for _ in range(N)]))]:
                found_state = False
                for _generated_state in states_N_2:
                    if _generated_state == _state:
                        found_state = True
                        break
                self.assertTrue(found_state)


if __name__ == '__main__':
    unittest.main()
