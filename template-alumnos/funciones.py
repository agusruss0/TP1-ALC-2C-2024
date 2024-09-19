import numpy as np
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


def solve_LU(L, U, b):
    y = sp.linalg.solve_triangular(L, b, lower=True)
    x = sp.linalg.solve_triangular(U, y)
    return x


def descompPLU(A):
    n = len(A)
    P = np.eye(n)
    L = np.eye(n)
    U = A.copy()

    for i in range(n - 1):
        # Buscar el índice del pivote máximo en la columna actual
        max_index = np.argmax(abs(U[i:, i])) + i

        if max_index != i:
            # Intercambiar filas en la matriz de permutación P
            P[[i, max_index]] = P[[max_index, i]]

            # Intercambiar filas en la matriz U
            U[[i, max_index]] = U[[max_index, i]]

            # Intercambiar filas en la matriz L (anterior)
            if i >= 1:
                L[[i, max_index], :i] = L[[max_index, i], :i]

        pivot = U[i, i]
        if pivot == 0:
            print("Tiene pivote nulo")
            return None

        L[i + 1 :, i] = U[i + 1 :, i] / pivot
        U[i + 1 :, i:] -= np.outer(L[i + 1 :, i], U[i, i:])

    return P, L, U


A = np.array([[1, -1, 0, 1], [0, 1, 4, 0], [2, -1, 0, -2], [-3, 3, 0, -1]])
rows, cols = A.shape
id = np.eye(rows, cols)

leontief = id - A
L, U, P = calcularPLU(leontief)
print(L)
print(U)
print(P)
print(L @ U)
print(np.linalg.inv(P))
A_permutada = L @ U
print(np.transpose(P) @ A_permutada)
print(leontief)
