import matplotlib.pyplot as plt
from matplotlib import ticker

from qubit_system.geometry import RegularLattice
from plots_creation.utils import save_current_fig

LATTICE_SPACING = 1.5e-6


def _plot_geometry(geometry: RegularLattice, ax):
    markersize = 8
    if geometry.dimensions <= 2:
        ax.plot(geometry.coordinates[:, 0], geometry.coordinates[:, 1], 'o', markersize=markersize)
        # for i, (x, y) in enumerate(geometry.coordinates):
        #     ax.text(x, y, i)

        ax.set_aspect('equal', 'datalim')
    elif geometry.dimensions == 3:
        ax.scatter(geometry.coordinates[:, 0], geometry.coordinates[:, 1], geometry.coordinates[:, 2],
                   s=markersize ** 2)
        # for i, (x, y, z) in enumerate(geometry.coordinates):
        #     ax.text(x, y, z, i)
        ax.zaxis.set_major_formatter(ticker.EngFormatter('m'))

    ax.locator_params(nbins=3, axis='both')
    ax.xaxis.set_major_formatter(ticker.EngFormatter('m'))
    ax.yaxis.set_major_formatter(ticker.EngFormatter('m'))

    ax.grid()


def plot_geometries():
    fig = plt.figure(figsize=(10, 2))
    ax = None
    for i, shape in enumerate([(8,), (4, 2), (2, 2, 2)]):
        if len(shape) == 3:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(131 + i, projection='3d')

            ax.text2D(0.95, 0.95, s=f"{len(shape)}D",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)
            ax.tick_params(axis='both', which='major', pad=5)
            ax.tick_params(axis='z', which='major', pad=15)
        else:
            if ax is not None:
                ax = fig.add_subplot(131 + i, sharey=ax)
            else:
                ax = fig.add_subplot(131 + i)
                # ax.set_xlim((-1e-6, 13e-6))
            ax.set_ylim((-5e-6, 7e-6))

            ax.text(0.95, 0.95, s=f"{len(shape)}D",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)


        geometry = RegularLattice(shape, spacing=LATTICE_SPACING)

        _plot_geometry(geometry, ax)
        # ax = fig.add_subplot(131 + i)

    plt.tight_layout()
    plt.subplots_adjust(
        left=0.08, right=0.92,
        bottom=0.12, top=0.95,
        wspace=0.3
    )
    save_current_fig("geometries")


if __name__ == '__main__':
    plot_geometries()
