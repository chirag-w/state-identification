import numpy as np

def density_matrix(pure_states):
    #Convert an array of pure states to their density matrix representation
    N = len(pure_states)
    S = []
    for i in range(N):
        S.append(np.outer(pure_states[i],pure_states[i].conjugate()))
    return S

def tensor_power(S,n):
    assert(n >= 1)
    rho = []
    N = len(S)
    for i in range(N):
        rho.append(S[i].copy())
        for j in range(n-1):
            rho[i] = np.kron(rho[i],S[i])
    return rho

def trace_distance(rho1,rho2):
    eig = np.abs(np.linalg.eigvalsh(rho1-rho2))
    return sum(eig)/2