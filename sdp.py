import picos as pcs
import numpy as np
from util import *
from scipy.linalg import sqrtm

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

def pretty_good_measurement(S, n = 1):
    S = tensor_power(S,n)

    N = len(S) #Number of states in S
    d = S[0].shape[0] #Dimension of states

    rho = np.zeros((d,d),dtype = complex)
    for i in range(N):
        rho += S[i]
    rho_inv = np.linalg.pinv(rho, hermitian=True)
    p = 0.0
    rho_inv_root = sqrtm(rho_inv)
    M = []
    for i in range(N):
        M.append(rho_inv_root @ S[i] @ rho_inv_root)
        p+= np.trace( M[i] @ S[i])/N
    p = p.real
    return p,M