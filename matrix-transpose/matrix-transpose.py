import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    R = len(A)
    C = len(A[0])
    T = [[0]*R for _ in range(C)]
    for i in range(R):
        for j in range (C):
            T[j][i] = A[i][j]
    Trans = np.array(T)
    return Trans
    pass
