from scipy import constants

_Delta = {
    4: {
        0: -12.6,
        0.04: -12.4,
        0.145: -15,
        0.4: 0,
        0.5: -1.3,
        0.77: 9.5,
        0.88: 9.3,
        1: 15,
        1.1: 20
    },
    8: {
        0: -15,
        0.21: -6.5,
        0.4: 0,
        0.53: 1.6,
        0.67: 5.9,
        0.82: 7.4,
        0.98: 12.8,
        1.1: 14.3
    }
}

Delta = {
    N:  {t * 1e-6: D * 2 * constants.pi * 1e6 for t, D in _Delta[N].items()}
    for N in _Delta.keys()
}


