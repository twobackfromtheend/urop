from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from qubit_system.geometry import BaseGeometry
from qubit_system.qubit_systems.decorators import plotting_decorator


class RegularLattice(BaseGeometry):
    def __init__(self, shape: Sequence[int], spacing: float = 1):
        super().__init__()
        self.spacing = spacing
        self.dimensions = len(shape)
        assert self.dimensions <= 3, "shape has to of length 3 or below (representing dimensions)"
        self.shape = shape

        self.coordinates = self._get_coordinates()

    def get_distance(self, i: int, j: int) -> float:
        return ((self.coordinates[j] - self.coordinates[i]) ** 2).sum() ** 0.5

    def _get_coordinates(self):
        coordinates = []

        if self.dimensions <= 2:
            if len(self.shape) == 1:
                shape = (self.shape[0], 1)
            else:
                shape = self.shape
            coordinates = []
            for x_coord in range(shape[0]):
                for y_coord in range(shape[1]):
                    coordinates.append((x_coord, y_coord))
        else:
            shape = self.shape
            for x_coord in range(shape[0]):
                for y_coord in range(shape[1]):
                    for z_coord in range(shape[2]):
                        coordinates.append((x_coord, y_coord, z_coord))
        return np.array(coordinates, dtype=np.float64) * self.spacing

    @plotting_decorator
    def plot(self):
        if self.dimensions <= 2:
            plt.figure()

            plt.plot(self.coordinates[:, 0], self.coordinates[:, 1], 'x')
            for i, (x, y) in enumerate(self.coordinates):
                plt.text(x, y, i)

                col = i // self.shape[1]
                row = i - self.shape[1] * col
                plt.text(x, y + self.spacing * 0.2, f"{row}, {col}", color='blue')

            plt.gca().set_aspect('equal', 'datalim')
            plt.grid()
            plt.tight_layout()
        elif self.dimensions == 3:
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2])
            for i, (x, y, z) in enumerate(self.coordinates):
                ax.text(x, y, z, i)

                x_i = i // (self.shape[1] * self.shape[2])
                y_i = (i - x_i * (self.shape[1] * self.shape[2])) // (self.shape[2])
                z_i = (i - x_i * (self.shape[1] * self.shape[2]) - y_i * self.shape[2])
                ax.text(x, y, z + self.spacing * 0.2, f"{x_i}, {y_i}, {z_i}", color='blue')

            plt.grid()
            plt.tight_layout()


if __name__ == '__main__':
    lattice = RegularLattice(shape=(4, 3, 2), spacing=1.5e-6, )
    # lattice = RegularLattice(shape=(4, 3), spacing=1.5e-6, )
    print(lattice.coordinates)

    print(lattice.get_distance(0, 3))

    lattice.plot(show=True)
