import numpy as np
from util import *
from sdp import *

def generate_pauli_eigenstates():
    #Return the density matrix representations of |0>,|1>,|+>,|->,|+i>,|-i>
    pure_states = [np.array([1,0],dtype = 'cdouble'),np.array([0,1],dtype = 'cdouble'),np.array([1/np.sqrt(2),1/np.sqrt(2)],dtype = 'cdouble'),
    np.array([1/np.sqrt(2),-1/np.sqrt(2)],dtype = 'cdouble'),np.array([1/np.sqrt(2),1j/np.sqrt(2)],dtype = 'cdouble'),np.array([1/np.sqrt(2),-1j/np.sqrt(2)],dtype = 'cdouble')
    ]
    return density_matrix(pure_states)

S = generate_pauli_eigenstates()
print("Set of states: ")
for i in range(6):
    print(S[i])
p,M = identification_probability(S)
print("State identification probability: ",p)
print("Optimal measurement observables: ")
for i in range(len(M)):
    print(M[i])