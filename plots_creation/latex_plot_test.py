import matplotlib.pyplot as plt

import plots_creation.utils

# import matplotlib.font_manager
# print([f.name for f in matplotlib.font_manager.fontManager.ttflist])

plt.figure()
# plt.title(r"$\ket{ \qa  \qb  \qa  \qb  \qb  \qb }$", usetex=False)
# plt.rcParams['font.serif'] = ['STIXGeneral']
plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['font.serif'] = ['Comic Sans MS']
plt.rcParams['font.family'] = 'serif'
plt.title(r"asdasd $◒◒◒ \rangle$", usetex=False, fontdict={'family': 'serif'})

plt.show()
