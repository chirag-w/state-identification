import numpy as np
from util import *
from sdp import *

def mub_prime_states(p):
    #Returns all states associated with p-dimensional MUBs where p is prime
    Z = np.diag(np.exp(2j*np.pi * np.arange(p)/p))
    X = np.zeros((p,p), dtype = complex)
    for i in range(p):
        X[(i+1)%p][i] = 1
    M = [Z]
    prod = X.copy()
    for i in range(p):
        M.append(prod)
        prod = prod@Z

    #States are eigenvectors of each matrix in M
    S = []
    for i in range(len(M)):
        _,v = np.linalg.eig(M[i])
        for col in range(p):
            S.append(v[:,col])
    return density_matrix(S)


S = mub_prime_states(3)
for i in range(len(S)):
    print(S[i])

for n in range(1,4):
    print(n, "copies:")
    p,M = pretty_good_measurement(S,n)
    print("PGM Success probability: ",p)
    p,M = state_identification(S,n)
    print("Optimal State identification probability: ",p)
    # print("Optimal measurement operators: ")
    # for i in range(len(M)):
    #     print(M[i])