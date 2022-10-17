import numpy as np
from util import *
from sdp import *

def MUB8_States():
    #Returns 9 mutually unbiased bases of dimension d = 8
    id = np.diag([1. for i in range(8)])

    M = np.array([[[1.+0j,1.],[1.,-1.]],[[1.,1.],[1j,-1j]]])/np.sqrt(2)
    unitaries = [id, np.diag([1.,1.,1.,1.,1.,-1.,-1.,1.]),np.diag([1.,1.,1.,-1.,1.,-1.,1.,1.]),np.diag([1.,1.,1.,-1.,1.,1.,-1.,1.])]
    preMUB = np.zeros((8,8,8),dtype = complex)
    for i in range(8):
        preMUB[i] = np.kron(np.kron(M[int(i/4)],M[int((i%4)/2)]),M[i%2])
    MUB = np.zeros((9,8,8), dtype = complex)
    MUB[0] = id
    for i in range(8):
        MUB[i+1] = unitaries[min(i,7-i)]@preMUB[i]
    
    S = []
    for i in range(9):
        for j in range(8):
            S.append(MUB[i,:,j])
    return density_matrix(S)

S = MUB8_States()

for n in range(1,2):
    p = info_theory_upper_bound(S,n)
    print("Information-theoretic upper bound on success probability: ",p)
    # print(n, "copies:")
    # p,M = state_identification(S,n)
    # print("State identification probability: ",p)
    # print("Optimal measurement operators: ")
    # for i in range(len(M)):
    #     print(M[i])
    #     #print(np.linalg.norm(M[i]-S[i]/9)) #Verify that M[i] = S[i]/9



