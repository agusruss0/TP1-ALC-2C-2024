import numpy as np  # type:ignore
import scipy as sp  # type:ignore


def calcularPLU(A):
    """Calcula L,U de una matriz A usando el metodo de pivoteo parcial."""
    rows, cols = A.shape
    L = np.eye(
        rows, cols
    )  # Inicializa L como matriz identidad (matriz triangular inferior)
    U = (
        A.copy()
    )  # U comienza como una copia de A (se convertirá en matriz triangular superior)
    P = np.eye(rows, cols)  # Inicializa P como matriz identidad
    for i in range(rows):
        k_pivot = i
        P_i = np.eye(rows, cols)  # Crea una matriz de permutación para cada iteración

        # Maneja el caso de ceros en la diagonal
        while U[i, i] == 0:
            if U[k_pivot, i] != 0:
                # Intercambia filas en U y P_i
                pivot = U[i, i:].copy()
                U[i, i:] = U[k_pivot, i:]
                U[k_pivot, i:] = pivot

                pivot_id = P_i[i, i:].copy()
                P_i[i, i:] = P_i[k_pivot, i:]
                P_i[k_pivot, i:] = pivot_id
            else:
                k_pivot += 1

        P = P_i @ P  # Actualiza la matriz de permutación total
        for j in range(i + 1, rows):
            # Calcula el factor de eliminación
            factor = U[j, i] / U[i, i]
            # Actualiza U: elimina elementos bajo la diagonal
            U[j, i:] = U[j, i:] - factor * U[i, i:]
            # Actualiza L: almacena los factores de eliminación
            L[j, i] = factor

    return L, U, P


def calcularLU(A):
    """
    Calcula la descomposición LU de una matriz A.

    Args:
    A (numpy.ndarray): Matriz de entrada.

    Returns:
    tuple: (L, U) donde L es la matriz triangular inferior y U es la matriz triangular superior.
    """
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
    """
    Calcula la inversa de una matriz utilizando su descomposición LU.

    Args:
    L (numpy.ndarray): Matriz triangular inferior.
    U (numpy.ndarray): Matriz triangular superior.

    Returns:
    numpy.ndarray: La matriz inversa.
    """
    rows, cols = L.shape
    Inv = np.zeros(L.shape)
    id = np.eye(rows, cols)
    for i in range(rows):
        Inv[:, i] = solve_LU(L, U, id[:, i])

    return Inv


def inversaPLU(L, U, P):
    """
    Calcula la inversa de una matriz utilizando su descomposición LU con pivoteo.

    Args:
    L (numpy.ndarray): Matriz triangular inferior.
    U (numpy.ndarray): Matriz triangular superior.
    P (numpy.ndarray): Matriz de permutación.

    Returns:
    numpy.ndarray: La matriz inversa.
    """
    return inversaLU(L, U) @ P


def solve_LU(L, U, b):
    """
    Resuelve el sistema de ecuaciones Ax = b utilizando la descomposición LU.

    Args:
    L (numpy.ndarray): Matriz triangular inferior.
    U (numpy.ndarray): Matriz triangular superior.
    b (numpy.ndarray): Vector del lado derecho del sistema.

    Returns:
    numpy.ndarray: La solución del sistema.
    """
    y = sp.linalg.solve_triangular(L, b, lower=True)
    x = sp.linalg.solve_triangular(U, y)
    return x


def calcular_coeficientes_tecnicos(Z, P):
    """
    Calcula los coeficientes técnicos utilizando la descomposición LU con pivoteo.

    Args:
    Z (numpy.ndarray): Matriz de insumo-producto.
    P (numpy.ndarray): Matriz de producción total.

    Returns:
    numpy.ndarray: Matriz de coeficientes técnicos.
    """
    return Z @ inversaPLU(*calcularPLU(P))
