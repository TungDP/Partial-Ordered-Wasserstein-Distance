def get_E_F(N, M, backend):
    nx = backend
    mid_para = nx.sqrt((1 / (N**2) + 1 / (M**2)))

    a_n = nx.arange(start=1, stop=N + 1)
    b_m = nx.arange(start=1, stop=M + 1)
    row_col_matrix = nx.meshgrid(a_n, b_m)  # indexing xy
    row = row_col_matrix[0].T / N  # row = (i+1)/N
    col = row_col_matrix[1].T / M  # col = (j+1)/M

    l = nx.abs(row - col) / mid_para

    E = 1 / ((row - col) ** 2 + 1)
    F = l**2
    return E, F
