from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.qubit_systems.decorators import plotting_decorator


class RegularLattice3D(BaseGeometry):
    def __init__(self, shape: Tuple[int, int, int], spacing: float = 1):
        super().__init__()
        self.spacing = spacing
        self.shape = shape

        coordinates = []

        for x_coord in range(shape[0]):
            for y_coord in range(shape[1]):
                for z_coord in range(shape[2]):
                    coordinates.append((x_coord, y_coord, z_coord))

        self.coordinates = np.array(coordinates, dtype=np.float64) * spacing

    def get_distance(self, i: int, j: int) -> float:
        return ((self.coordinates[j] - self.coordinates[i]) ** 2).sum() ** 0.5

    @plotting_decorator
    def plot(self):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2])
        for i, (x, y, z) in enumerate(self.coordinates):
            ax.text(x, y, z, i)

        plt.grid()
        plt.tight_layout()

    def __hash__(self):
        return hash((self.__class__.__name__, self.spacing))


if __name__ == '__main__':
    lattice = RegularLattice3D(shape=(3, 3, 3), spacing=0.5)

    print(lattice.coordinates)

    print(lattice.get_distance(0, 5))

    lattice.plot(show=True)
