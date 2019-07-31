
def get_C6(nRyd: int):
    C6_GHz_micm2au = 1.0 / 1.448e-19

    c6_au_ns = -(nRyd ** 11) * (11.97 - 0.8486 * nRyd + 3.385e-3 * nRyd * nRyd) / C6_GHz_micm2au * 1e3
    # [MHz [micrometers]^6]

    return c6_au_ns * 1e-30  # Hz m^6
