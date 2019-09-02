from qubit_system.geometry.base_geometry import BaseGeometry
import matplotlib.pyplot as plt

from qubit_system.qubit_system_classes import plotting_decorator


class RegularLattice1D(BaseGeometry):

    def __init__(self, spacing: float = 1):
        super().__init__()
        self.spacing = spacing

    def get_distance(self, i: int, j: int) -> float:
        return abs(j - i) * self.spacing

    @plotting_decorator
    def plot(self):
        plt.figure()

        POINTS_TO_PLOT = 3
        _x = [self.spacing * i for i in range(POINTS_TO_PLOT)]
        _y = [0 for _ in range(POINTS_TO_PLOT)]
        plt.plot(_x, _y, 'x')

        for i in range(POINTS_TO_PLOT):
            plt.text(_x[i], _y[i], i)

        plt.text(_x[-1] + self.spacing, _y[-1], "...")
        plt.grid()
        plt.tight_layout()

    def __hash__(self):
        return hash((self.__class__.__name__, self.spacing))


if __name__ == '__main__':
    lattice = RegularLattice1D(spacing=0.5)

    lattice.plot(show=True)
