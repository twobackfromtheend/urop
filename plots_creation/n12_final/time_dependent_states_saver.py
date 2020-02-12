import os

from plots_creation.n12_final.plots_creation_3 import save_time_dependent_energies

# BO_FILES = [
#     f"12_BO_COMPARE_BO_3D_std_",
#     f"12_BO_COMPARE_BO_3D_alt_",
#     f"12_BO_COMPARE_BO_2D_std_",
#     f"12_BO_COMPARE_BO_2D_alt_",
#     f"12_BO_COMPARE_BO_1D_std_",
#     f"12_BO_COMPARE_BO_1D_alt_",
# ]

BO_FILE = os.getenv("BO_FILE")
save_time_dependent_energies([BO_FILE])
