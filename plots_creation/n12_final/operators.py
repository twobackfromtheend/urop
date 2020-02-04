import quimb as q


def get_total_kinks_opt(N):
    total_kinks_opt = 0
    for i in range(N - 1):
        sigmaz_i = q.ikron(q.pauli("z"), [2] * N, i, sparse=False)
        sigmaz_i1 = q.ikron(q.pauli("z"), [2] * N, i + 1, sparse=False)
        total_kinks_opt += (1 - sigmaz_i @ sigmaz_i1) / 2
    total_kinks_opt = q.qu(total_kinks_opt, sparse=False)
    return total_kinks_opt


def get_total_M_opt(N):
    total_M_opt = 0
    for i in range(N):
        total_M_opt += q.ikron(q.pauli("z"), [2] * N, i, sparse=True)
    total_M_opt = q.qu(total_M_opt, sparse=False)
    return total_M_opt


if __name__ == '__main__':
    N = 6
    kinks_opt = get_total_kinks_opt(N)

    e = q.ket([1, 0])
    g = q.ket([0, 1])
    all_e = q.qu(q.kron(*[e for _ in range(N)]), sparse=False)
    all_g = q.qu(q.kron(*[g for _ in range(N)]), sparse=False)
    print(q.expec(kinks_opt, all_e))
    print(q.expec(kinks_opt, all_g))
    ghz = q.normalize(all_e + all_g)

    print(q.expec(kinks_opt, ghz))

    egeg = q.qu(q.kron(*[e, g, e, g, e, g]), sparse=False)
    gege = q.qu(q.kron(*[g, e, g, e, g, e]), sparse=False)
    print(q.expec(kinks_opt, egeg))
    print(q.expec(kinks_opt, gege))
    ghz_alt = q.normalize(egeg + gege)
    print(q.expec(kinks_opt, ghz_alt))
