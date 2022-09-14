import numpy as np
from util import *
from sdp import *
import matplotlib
from matplotlib import pyplot as plt 

def RX(theta):
    m = np.zeros((2,2), dtype = complex)
    m[0,0] = m[1,1] = np.cos(theta/2)
    m[0,1] = m[1,0] = -np.sin(theta/2)*1j
    return m
def gen_states(n):
    theta = 2*np.pi/n
    R = RX(theta)
    cur_state = np.array([1,0],dtype = complex)
    S = []
    for i in range(n):
        S.append(cur_state)
        cur_state = R@cur_state
    return density_matrix(S)

eps = []
num_states = []
tr_dist = []
prob = []

N = 6
n = 1
S = gen_states(N)
rho = tensor_power(S,n)
eps.append(np.trace(rho[0]@rho[1]))
tr_dist.append(trace_distance(rho[0],rho[1]))
num_states.append(N)

print(num_states)
print(eps)
print(tr_dist)

p,M = pretty_good_measurement(S,n)
#p,M = state_identification(S,n)
prob.append(p)
print(prob)
# for i in range(N):
#     print('State ',i+1,':')
#     print(rho[i])
#     print('Operator ',i+1,':')
#     print(M[i]-p*rho[i])