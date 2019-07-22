from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.qubit_system_classes import plotting_decorator


class RegularLattice2D(BaseGeometry):
    def __init__(self, shape: Tuple[int, int], spacing: float = 1):
        super().__init__()
        self.spacing = spacing
        self.shape = shape

        coordinates = []

        for x_coord in range(shape[0]):
            for y_coord in range(shape[1]):
                coordinates.append((x_coord, y_coord))

        self.coordinates = np.array(coordinates, dtype=np.float64) * spacing

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


if __name__ == '__main__':
    lattice = RegularLattice2D(shape=(3, 3), spacing=0.5)

    print(lattice.coordinates)

    print(lattice.get_distance(0, 5))

    lattice.plot(show=True)
