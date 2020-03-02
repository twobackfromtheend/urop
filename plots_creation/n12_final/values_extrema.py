import numpy as np
from scipy.signal import argrelextrema

from optimised_protocols import saver

BO_FILES = [
    f"12_BO_COMPARE_BO_3D_std_",
    f"12_BO_COMPARE_BO_3D_alt_",
    f"12_BO_COMPARE_BO_2D_std_",
    f"12_BO_COMPARE_BO_2D_alt_",
    # f"12_BO_COMPARE_BO_1D_std_",
    # f"12_BO_COMPARE_BO_1D_alt_",

    f"12_BO_COMPARE_BO_WIDER_1D_std_",
    f"12_BO_COMPARE_BO_WIDER_1D_alt_",
]

for BO_file in BO_FILES:
    print(BO_file)
    e_qs = saver.load(BO_file, solve=False)
    Omega = e_qs.Omega
    t_list = e_qs.t_list
    local_maxima_i = argrelextrema(Omega, np.greater)[0]
    print(local_maxima_i)
    for i in local_maxima_i:
        print(f"Omega: {Omega[i]:.5e}")
        print(f"t: {t_list[i]:.5e}")
