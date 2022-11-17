from multiprocessing.util import info
from turtle import pu
import numpy as np
from util import *
from sdp import *

def generate_pauli_eigenstates():
    #Return the density matrix representations of |0>,|1>,|+>,|->,|+i>,|-i>
    pure_states = [np.array([1,0],dtype = complex),np.array([0,1],dtype = complex),np.array([1/np.sqrt(2),1/np.sqrt(2)],dtype = complex),
    np.array([1/np.sqrt(2),-1/np.sqrt(2)],dtype = complex),np.array([1/np.sqrt(2),1j/np.sqrt(2)],dtype = complex),np.array([1/np.sqrt(2),-1j/np.sqrt(2)],dtype = complex)
    ]
    return density_matrix(pure_states)

def generate_two_qubit_states():
    single_states = [np.array([1,0],dtype = complex),np.array([0,1],dtype = complex),np.array([1/np.sqrt(2),1/np.sqrt(2)],dtype = complex),
    np.array([1/np.sqrt(2),-1/np.sqrt(2)],dtype = complex),np.array([1/np.sqrt(2),1j/np.sqrt(2)],dtype = complex),np.array([1/np.sqrt(2),-1j/np.sqrt(2)],dtype = complex)
    ]
    pure_states = []
    for i in range(6):
        for j in range(6):
            if j == i:
                continue
            pure_states.append(np.kron(single_states[i],single_states[j]))
    return density_matrix(pure_states)


S = generate_two_qubit_states()
# print("Set of states: ")
# for i in range(len(S)):
#     print(S[i])

for n in range(1,2):
    print(n, "copies:")
    p,M = pretty_good_measurement(S,n)
    print("Pretty good measurement success probability: ",p)
    p = info_theory_upper_bound(S,n)
    print("Information-theoretic upper bound on success probability: ",p)
    p,M = state_identification(S,n)
    print("State identification probability: ",p)
    # print("Optimal measurement operators: ")
    # for i in range(len(M)):
    #     print(M[i])