import numpy as np
import scipy as sp

def calcularLU(A):
    rows, cols = A.shape
    L = np.eye(rows)
    U = A.copy()

    if rows != cols:
        raise ValueError("La matriz debe ser cuadrada para realizar la descomposición LU.")
    
    for i in range(rows):
        for j in range(i+1,rows):
            factor = U[j,i]/U[i,i]                           #Calculo el factor por el cual modifico la fila a escalonar
            U[j,i:] = U[j,i:] - factor * U[i,i:]             #Le resto a la fila j de U la fila i multiplicada por el factor que la escalona
            L[j,i] = factor                                  #Armo L que es los factores como columnas de bajo de la diagonal con el signo opuesto
           
    return L, U

def inversaLU(L, U):
    rows, cols = L.shape
    Inv = np.zeros(L.shape)
    id = np.eye(rows, cols)
    for i in range(rows):
        Inv[:,i] = solve_LU(L,U, id[:,i])

    return Inv

def solve_LU(L, U, b):
    y = sp.linalg.solve_triangular(L,b, lower=True)
    x = sp.linalg.solve_triangular(U, y)
    return x

A = np.array([[1,-1,0,1],[0,1,4,0],[2,-1,0,-2],[-3,3,0,-1]])

print(inversaLU(*calcularLU(A)))
