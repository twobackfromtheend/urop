import logging
from pathlib import Path

import numpy as np

from qubit_system.qubit_systems.ti_evolving_qubit_system import TimeIndependentEvolvingQubitSystem


logger = logging.getLogger(__name__)

if __name__ == '__main__':
    from qubit_system.geometry.regular_lattice_1d import RegularLattice1D

    # s_qs = StaticQubitSystem(
    #     N=4, V=1,
    #     geometry=RegularLattice1D(),
    #     Omega=1, Delta=np.linspace(-1, 1, 50)
    # )
    # s_qs.plot()

    from qubit_system.utils.ghz_states import StandardGHZState

    t = 1
    N = 20
    e_qs = TimeIndependentEvolvingQubitSystem(
        N=N, V=1, geometry=RegularLattice1D(),
        Omega=1, Delta=1,
        t_list=np.linspace(0, 1, 20),
        ghz_state=StandardGHZState(N)
    )
    e_qs.solve()
    e_qs.plot(with_antisymmetric_ghz=True, show=True)

    # t = 1
    # N = 4
    # t_num = 30
    # e_qs = EvolvingQubitSystem(
    #     N=N, V=1, geometry=RegularLattice1D(),
    #     Omega=np.ones(t_num - 1),
    #     Delta=np.linspace(-1, 1, t_num - 1),
    #     t_list=np.linspace(0, 1, t_num),
    #     ghz_state=StandardGHZState(N),
    #     solve_points_per_timestep=50
    # )
    # e_qs.solve()
    # e_qs.plot(with_antisymmetric_ghz=True, show=True)
