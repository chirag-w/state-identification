import numpy as np
from util import *
from sdp import *

def MUB4_States():
    #Returns the 20 states associated with the 4-dimensional MUB
    B = np.zeros((20,4),dtype = complex)
    
    B[0] =  np.array([1,0,0,0])
    B[1] =  np.array([0,1,0,0])
    B[2] =  np.array([0,0,1,0])
    B[3] =  np.array([0,0,0,1])

    B[4] = 0.5*np.array([1,1,1,1])
    B[5] = 0.5*np.array([1,1,-1,-1])
    B[6] = 0.5*np.array([1,-1,-1,1])
    B[7] = 0.5*np.array([1,-1,1,-1])

    B[8] = 0.5*np.array([1,-1,-1j,-1j])
    B[9] = 0.5*np.array([1,-1,1j,1j])
    B[10] = 0.5*np.array([1,1,1j,-1j])
    B[11] = 0.5*np.array([1,1,-1j,1j])

    B[12] = 0.5*np.array([1,-1j,-1j,-1])
    B[13] = 0.5*np.array([1,-1j,1j,1])
    B[14] = 0.5*np.array([1,1j,1j,-1])
    B[15] = 0.5*np.array([1,1j,-1j,1])

    B[16] = 0.5*np.array([1,-1j,-1,-1j])
    B[17] = 0.5*np.array([1,-1j,1,1j])
    B[18] = 0.5*np.array([1,1j,-1,1j])
    B[19] = 0.5*np.array([1,1j,1,-1j])

    return density_matrix(B)

S = MUB4_States()
print("Set of states: ")
for i in range(20):
    print(S[i])

for n in range(1,3):
    print(n, "copies:")
    p,M = pretty_good_measurement(S,n)
    print("Pretty good measurement success probability: ",p)
    p,M = state_identification(S,n)
    print("Optimal state identification probability: ",p)
    # print("Optimal measurement operators: ")
    # for i in range(len(M)):
    #     print(M[i])