import os
import re
from qubit_system.geometry import *

from qubit_system.utils.ghz_states import *

LATTICE_SPACING = 1.5e-6


def get_geometry_and_ghz_state(geometry_envvar: str, ghz_state_envvar: str):
    try:
        N = int(os.getenv("N"))
        return eval(geometry_envvar), eval(ghz_state_envvar)
    except Exception:
        match = re.match(r"^(\d+)_(\d)d_(std|alt)$", ghz_state_envvar)
        if match:
            N = int(match.group(1))
            D = int(match.group(2))
            ghz = match.group(3)
            if N == 12:
                if D == 1:
                    shape = (12,)
                    if ghz == "std":
                        ghz_single_component = [True, True, True, True, True, True, True, True, True, True, True, True]
                    elif ghz == "alt":
                        ghz_single_component = [True, False, True, False, True, False, True, False, True, False, True,
                                                False]
                elif D == 2:
                    shape = (4, 3)
                    if ghz == "std":
                        ghz_single_component = [True, True, True, True, True, True, True, True, True, True, True, True]
                    elif ghz == "alt":
                        ghz_single_component = [True, False, True, False, True, False, True, False, True, False, True,
                                                False]
                elif D == 3:
                    shape = (3, 2, 2)
                    if ghz == "std":
                        ghz_single_component = [True, True, True, True, True, True, True, True, True, True, True, True]
                    elif ghz == "alt":
                        ghz_single_component = [True, False, False, True, False, True, True, False, True, False, False,
                                                True]
                else:
                    raise ValueError(f"Unhandled ghz_state_envvar D: {D}")
                return RegularLattice(shape, spacing=LATTICE_SPACING), CustomGHZState(N, ghz_single_component)
            else:
                raise ValueError(f"Unhandled ghz_state_envvar N: {N}")

        else:
            raise ValueError(f"Unknown ghz_state_envvar: {ghz_state_envvar}")


if __name__ == '__main__':
    print(get_geometry_and_ghz_state("", "12_1d_alt"))
