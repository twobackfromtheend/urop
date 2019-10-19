from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from qubit_system.geometry.base_noisy_geometry import BaseNoisyGeometry
from qubit_system.qubit_systems.decorators import plotting_decorator


class NoisyRegularLattice(BaseNoisyGeometry):
    def __init__(self, shape: Sequence[int], spacing: float = 1, spacing_noise: float = 0):
        super().__init__()
        self.spacing = spacing
        self.spacing_noise = spacing_noise
        self.dimensions = len(shape)
        assert self.dimensions <= 3, "shape has to of length 3 or below (representing dimensions)"
        self.shape = shape

        self.base_coordinates = self._get_base_coordinates()
        self.coordinates = self._generate_coordinates()

    def get_distance(self, i: int, j: int, with_noise: bool = False) -> float:
        if with_noise:
            return ((self.coordinates[j] - self.coordinates[i]) ** 2).sum() ** 0.5
        else:
            return ((self.base_coordinates[j] - self.base_coordinates[i]) ** 2).sum() ** 0.5

    def _get_base_coordinates(self):
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

    def _generate_coordinates(self):
        return self.base_coordinates + np.random.normal(scale=self.spacing_noise, size=self.base_coordinates.shape)

    @plotting_decorator
    def plot(self):
        if self.dimensions <= 2:
            plt.figure()

            plt.plot(self.coordinates[:, 0], self.coordinates[:, 1], 'x')
            plt.plot(self.base_coordinates[:, 0], self.base_coordinates[:, 1], 'x', alpha=0.3)
            for i, (x, y) in enumerate(self.coordinates):
                plt.text(x, y, i)

            plt.gca().set_aspect('equal', 'datalim')
            plt.grid()
            plt.tight_layout()
        elif self.dimensions == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2])
            for i, (x, y, z) in enumerate(self.coordinates):
                ax.text(x, y, z, i)

            plt.grid()
            plt.tight_layout()


if __name__ == '__main__':
    lattice = NoisyRegularLattice(shape=(4, 2), spacing=1.5e-6, spacing_noise=5e-9)
    print(lattice.coordinates)

    print(lattice.get_distance(0, 3))

    lattice.plot(show=True)

    for i in range(5):
        lattice.reset_coordinates()
        lattice.plot(show=True)
