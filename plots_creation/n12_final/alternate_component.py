

def get_alternate_component(N: int, D: int):
    if N == 8:
        if D == 1:
            ghz_single_component = [True, False, True, False, True, False, True, False]
            shape = (8,)
        elif D == 2:
            ghz_single_component = [True, False, False, True, True, False, False, True]
            shape = (4, 2)
        elif D == 3:
            ghz_single_component = [True, False, False, True, False, True, True, False]
            shape = (2, 2, 2)
    elif N == 12:
        if D == 1:
            ghz_single_component = [True, False, True, False, True, False, True, False, True, False, True, False]
            shape = (12,)
        elif D == 2:
            ghz_single_component = [True, False, True, False, True, False, True, False, True, False, True, False]
            shape = (4, 3)
        elif D == 3:
            ghz_single_component = [True, False, False, True, False, True, True, False, True, False, False, True]
            shape = (3, 2, 2)
    elif N == 16:
        if D == 1:
            ghz_single_component = [True, False, True, False, True, False, True, False, True, False, True, False,
                                    True, False, True, False]
            shape = (16,)
        elif D == 2:
            ghz_single_component = [True, False, True, False, False, True, False, True, True, False, True, False,
                                    False, True, False, True]
            shape = (4, 4)
        elif D == 3:
            ghz_single_component = [True, False, False, True, False, True, True, False, True, False, False, True,
                                    False, True, True, False]
            shape = (4, 2, 2)

    try:
        return ghz_single_component, shape
    except:
        raise ValueError(f"Unknown N and D: {N}, {D}")
