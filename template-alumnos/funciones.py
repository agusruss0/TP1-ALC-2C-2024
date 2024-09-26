import numpy as np  # type:ignore
import scipy as sp  # type:ignore


def calcularPLU(A):
    rows, cols = A.shape
    L = np.eye(rows, cols)
    U = A.copy()
    P = np.eye(rows, cols)
    for i in range(rows):
        k_pivot = 1
        P_i = np.eye(rows, cols)
        while U[i, i] == 0:
            if U[k_pivot, i] != 0:
                pivot = U[i, i:].copy()
                U[i, i:] = U[k_pivot, i:]
                U[k_pivot, i:] = pivot

                pivot_id = P_i[i, i:].copy()
                P_i[i, i:] = P_i[k_pivot, i:]
                P_i[k_pivot, i:] = pivot_id
            else:
                k_pivot += 1

        P = P_i @ P
        for j in range(i + 1, rows):
            factor = U[j, i] / U[i, i]
            U[j, i:] = U[j, i:] - factor * U[i, i:]
            L[j, i] = factor

    return L, U, P


def calcularLU(A):
    rows, cols = A.shape
    L = np.eye(rows, cols)
    U = A.copy()

    for i in range(rows):
        for j in range(i + 1, rows):
            factor = U[j, i] / U[i, i]
            U[j, i:] = U[j, i:] - factor * U[i, i:]
            L[j, i] = factor

    return L, U


def inversaLU(L, U):
    rows, cols = L.shape
    Inv = np.zeros(L.shape)
    id = np.eye(rows, cols)
    for i in range(rows):
        Inv[:, i] = solve_LU(L, U, id[:, i])

    return Inv


def inversaLUpivot(L, U, P):
    return inversaLU(L, U) @ P


def solve_LU(L, U, b):
    y = sp.linalg.solve_triangular(L, b, lower=True)
    x = sp.linalg.solve_triangular(U, y)
    return x


def calcular_coeficientes_tecnicos(Z, P):
    return Z @ inversaLUpivot(*calcularPLU(P))
