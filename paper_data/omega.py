from scipy import constants

_Omega = {
    4: {
        0: 0,
        0.01: 1,
        0.23: 5,
        0.4: 4,
        0.48: 4.81,
        0.52: 4.7,
        0.6: 5,
        0.8: 5,
        1.1: 0
    },
    8: {
        0: 0,
        0.05: 0.23,
        0.14: 2.72,
        0.23: 4.4,
        0.3: 4.8,
        0.48: 3.4,
        0.62: 5,
        0.71: 5,
        0.8: 4.4,
        0.9: 4,
        1.1: 0
    }
}

Omega = {
    N:  {t * 1e-6: O * 2 * constants.pi * 1e6 for t, O in _Omega[N].items()}
    for N in _Omega.keys()
}

