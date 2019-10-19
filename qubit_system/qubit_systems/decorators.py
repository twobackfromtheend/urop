from pathlib import Path

from matplotlib import pyplot as plt

PLOT_FOLDER = Path(__file__).parent.parent / 'plots'
PLOT_FOLDER.mkdir(exist_ok=True)


def plotting_decorator(func):
    def wrapper(*args, show: bool = False, savefig_name: str = None, **kwargs):
        return_value = func(*args, **kwargs)

        if savefig_name:
            plt.savefig(PLOT_FOLDER / savefig_name, dpi=300)
        if show:
            plt.show()

        return return_value

    return wrapper
