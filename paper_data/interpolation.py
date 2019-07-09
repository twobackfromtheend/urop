from scipy.interpolate import CubicSpline


def get_interpolated_function(_data_dict: dict):
    return CubicSpline(list(_data_dict.keys()), list(_data_dict.values()))


if __name__ == '__main__':
    from paper_data import Delta
    import numpy as np
    import matplotlib.pyplot as plt

    d = Delta[4]
    cs = get_interpolated_function(d)

    _x = np.linspace(min(d.keys()), max(d.keys()), 1000)

    plt.plot(list(d.keys()), list(d.values()), 'k')

    plt.plot(_x, cs(_x), '--')

    plt.show()
