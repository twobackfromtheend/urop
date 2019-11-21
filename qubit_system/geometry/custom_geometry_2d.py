import matplotlib.pyplot as plt
import numpy as np

from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.qubit_systems.decorators import plotting_decorator


class CustomGeometry2D(BaseGeometry):
    def __init__(self, coordinates: np.ndarray, scaling: float = 1):
        super().__init__()
        self.scaling = scaling
        self.coordinates = coordinates

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
    sin60 = (3 ** 0.5) / 2
    coordinates = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [0.5, sin60],
        [1.5, sin60],
        [0, 2 * sin60],
        [1, 2 * sin60],
        [2, 2 * sin60],
    ])
    geometry = CustomGeometry2D(coordinates=coordinates)

    print(geometry.coordinates)

    print(geometry.get_distance(0, 5))

    geometry.plot(show=True)
