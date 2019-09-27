from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.qubit_system_classes import plotting_decorator


class Star(BaseGeometry):
    """
    Equivalent to DoubleRing with the inner ring offset by angular_separation_between_qubits / 2
    """
    def __init__(self, N: int, spacing: float = 1):
        super().__init__()

        rings = 2

        assert N % rings == 0, f"N has to be multiple of {rings}, not {N}"

        N_per_ring = int(N / rings)
        self.spacing = spacing

        angular_separation_between_qubits = 2 * np.pi / N_per_ring
        unit_inner_ring_coordinates = []
        unit_outer_ring_coordinates = []
        for i in range(N_per_ring):
            inner_theta = angular_separation_between_qubits * i + angular_separation_between_qubits / 2
            unit_inner_ring_coordinates.append((np.cos(inner_theta), np.sin(inner_theta)))
            outer_theta = angular_separation_between_qubits * i
            unit_outer_ring_coordinates.append((np.cos(outer_theta), np.sin(outer_theta)))
        unit_inner_ring_coordinates = np.array(unit_inner_ring_coordinates)
        unit_outer_ring_coordinates = np.array(unit_outer_ring_coordinates)

        distance_between_qubits = np.sum((unit_inner_ring_coordinates[0] - unit_inner_ring_coordinates[1]) ** 2) ** 0.5

        scaling_factor = spacing / distance_between_qubits

        coordinates = []
        for i in range(rings):
            _scaling_factor = scaling_factor * (i + 1)
            if i == 0:
                coordinates.append(unit_inner_ring_coordinates * _scaling_factor)
            else:
                coordinates.append(unit_outer_ring_coordinates * _scaling_factor)

        self.coordinates = np.vstack(coordinates)

    def get_distance(self, i: int, j: int) -> float:
        return ((self.coordinates[j] - self.coordinates[i]) ** 2).sum() ** 0.5

    @plotting_decorator
    def plot(self):
        plt.figure()

        plt.plot(self.coordinates[:, 0], self.coordinates[:, 1], 'x')
        for i, (x, y) in enumerate(self.coordinates):
            plt.text(x, y, i)

        plt.grid()
        plt.tight_layout()

    def __hash__(self):
        return hash((self.__class__.__name__, self.spacing))


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    geometry = Star(8, spacing=1)
    # geometry.plot(show=True,)
    geometry.plot(show=True, savefig_name="geometry_star_8")
