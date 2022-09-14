import numpy as np
from util import *
from sdp import *

def generate_pauli_eigenstates():
    #Return the density matrix representations of |0>,|1>,|+>,|->,|+i>,|-i>
    pure_states = [np.array([1,0],dtype = complex),np.array([0,1],dtype = complex),np.array([1/np.sqrt(2),1/np.sqrt(2)],dtype = complex),
    np.array([1/np.sqrt(2),-1/np.sqrt(2)],dtype = complex),np.array([1/np.sqrt(2),1j/np.sqrt(2)],dtype = complex),np.array([1/np.sqrt(2),-1j/np.sqrt(2)],dtype = complex)
    ]
    return density_matrix(pure_states)

S = generate_pauli_eigenstates()
print("Set of states: ")
for i in range(4):
    print(S[i])

for n in range(1,6):
    print(n, "copies:")
    p,M = pretty_good_measurement(S,n)
    print("Pretty good measurement success probability: ",p)
    p,M = state_identification(S,n)
    print("State identification probability: ",p)
    # print("Optimal measurement operators: ")
    # for i in range(len(M)):
    #     print(M[i])