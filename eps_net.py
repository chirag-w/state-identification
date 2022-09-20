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



N = 3
S = gen_states(N)
overlap = np.trace(S[0]@S[1])
tr_dist = trace_distance(S[0],S[1])

print(N," states")
print("epsilon = ",tr_dist)
print("overlap = ",overlap)
for n in range(1,6):
    print(n," copies")
    p_pgm,_ = pretty_good_measurement(S,n)
    print("PGM success probability: ",p_pgm)
    try:
        p_opt,_ = state_identification(S,n)
        print("Optimal success probability: ",p_opt)
    except:
        print("Could not optimize for n = ",n)