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

for n in range(2,7):
    S = gen_states(n)
    eps.append(np.trace(S[0]@S[1]))
    tr_dist.append(trace_distance(S[0],S[1]))
    num_states.append(n)
    p,_ = state_identification(S)
    prob.append(p)
print(prob)