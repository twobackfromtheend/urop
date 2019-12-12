from pathlib import Path

from matplotlib import pyplot as plt

PAPER_PLOTS_FOLDER = "bo_paper_plots"

paper_plots_folder_path = Path(__file__).parent / PAPER_PLOTS_FOLDER
paper_plots_folder_path.mkdir(exist_ok=True)

LATEX_PREAMBLE = r"""
\usepackage{upgreek}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{braket}

\newcommand{\ghzalt}{\ket{\mathrm{GHZ}_8^\mathrm{alt}}}
\newcommand{\ghzstd}{\ket{\mathrm{GHZ}_8^\mathrm{std}}}

\newcommand{\qg}{{\circ}}
\newcommand{\qe}{{\bullet}}
"""

r"""
\usepackage{graphicx}
\usepackage{wasysym}
\usepackage{rotating}

\newcommand{\qa}{\mathpunct{\raisebox{.15\height}{\scalebox{0.7}{\rotatebox[origin=c]{90}{$\LEFTcircle$}}}}}
\newcommand{\qb}{\mathpunct{\begin{sideways}{$abc$}\end{sideways}}}

"""
# % \newcommand{\qa}{\mathpunct{\raisebox{.15\height}{\scalebox{0.7}{\rotatebox[origin=c]{90}{$\LEFTcircle$}}}}}
# % \newcommand{\qb}{\mathpunct{\raisebox{.15\height}{\scalebox{0.7}{\rotatebox[origin=c]{90}{$\RIGHTcircle$}}}}}
plt.rc('text', usetex=True)
plt.rc('font', family="serif", serif="CMU Serif")
plt.rc('text.latex', preamble=LATEX_PREAMBLE)


def save_current_fig(name: str):
    plt.savefig(
        paper_plots_folder_path / f"{name}.png",
        dpi=500
    )
    plt.close('all')

def _get_ghz_single_component_from_dimension(D: int, alt: bool = False):
    if not alt:
        return [True, True, True, True, True, True, True, True]
    if D == 1:
        return [True, False, True, False, True, False, True, False]
    elif D == 2:
        return [True, False, False, True, True, False, False, True]
    elif D == 3:
        return [True, False, False, True, False, True, True, False]
    else:
        raise ValueError(f"Unknown dimension: {D}")
