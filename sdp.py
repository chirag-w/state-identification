import picos as pcs
import numpy as np
from util import *

def state_identification(S,n = 1):
    #Return the maximum success probability for single-copy state identification
    #S = {rho_1,rho_2,...,rho_N}
    
    S = tensor_power(S,n)

    N = len(S) #Number of states in S
    d = S[0].shape[0] #Dimension of states

    sdp = pcs.Problem()

    I = pcs.Constant(np.diag([1. for i in range(d)]))

    M = [] #POVM Operators
    for i in range(N):
        M.append(pcs.HermitianVariable('M'+str(i+1),d))
        sdp.add_constraint(M[i] >> 0)
        sdp.add_constraint(M[i] << I)

    sdp.add_constraint(pcs.sum(M) == I)

    obj = 0.
    for i in range(N):
        obj+= pcs.trace(M[i]*S[i])/N
    sdp.set_objective('max',obj) 

    sol = sdp.solve(solver='cvxopt',verbosity=0)
    opt_M = []
    for i in range(N):
        opt_M.append(np.array(M[i].value, dtype = complex))
    return float(obj.real),opt_M


