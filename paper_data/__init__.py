from scipy import constants

from paper_data.delta import Delta
from paper_data.interpolation import get_interpolated_function
from paper_data.omega import Omega

V = 2 * constants.pi * 24e6

__all__ = [
    'V', 'Delta', 'Omega', 'get_interpolated_function'
]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    N = 4
    plt.plot(list(Delta[4].keys()), np.array(list(Delta[4].values())) / 2 / constants.pi)
    plt.xticks((0, 0.5e-6, 1.0e-6))
    plt.yticks((-15e6, 0, 15e6))

    plt.figure()
    plt.plot(list(Omega[4].keys()), np.array(list(Omega[4].values())) / 2 / constants.pi)
    plt.xticks((0, 0.5e-6, 1.0e-6))
    plt.yticks((0, 2.5e6, 5e6))

    plt.show()
