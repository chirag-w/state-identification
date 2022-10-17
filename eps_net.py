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

N_min = 2
N_max = 6
n_min = 1
n_max = 1

prob = np.zeros((N_max+1,n_max+1),dtype = float)
eps = np.zeros(N_max+1,dtype = float)

for N in range(N_min,N_max+1):
    S = gen_states(N)
    overlap = np.trace(S[0]@S[1])
    tr_dist = trace_distance(S[0],S[1])
    eps[N] = tr_dist
    print(N," states")
    print("epsilon = ",tr_dist)
    print("overlap = ",overlap)
    for n in range(n_min,n_max+1):
        print(n," copies")
        p_pgm,_ = pretty_good_measurement(S,n)
        print("PGM success probability: ",p_pgm)
        p = info_theory_upper_bound(S,n)
        print("Information-theoretic upper bound on success probability: ",p)
        # try:
        #     p_opt,_ = state_identification(S,n)
        #     print("Optimal success probability: ",p_opt)
        # except:
        #     print("Could not optimize for n = ",n)
        # prob[N,n] = p_pgm

# f1 = plt.figure()
# for N in range(N_min,N_max+1):
#     plt.plot(np.arange(n_min,n_max+1),prob[N,n_min:n_max+1], label = 'eps = '+str(eps[N]))
# plt.title('')
# plt.xlabel('Number of copies')    
# plt.ylabel('Success probability')
# plt.legend(loc = 'lower right')
# plt.show()

# f2 = plt.figure()
# col = ['blue','orange','green','red','black']
# for n in range(n_min,n_max+1):
#     plt.plot(eps[N_min:N_max+1],prob[N_min:N_max+1,n], label = 'n = '+str(n), color = col[n%5])
#     coeff = quad_upper_bound(prob[N_min:N_max+1,n],eps[N_min:N_max+1])
#     plt.plot(eps[N_min:N_max+1],coeff[0]*(eps[N_min:N_max+1]**2)+coeff[1]*eps[N_min:N_max+1]+coeff[2],linestyle = 'dashed', color = col[n%5], label = 'p = '+str(int(coeff[2]*1000)/1000)+'*e^2+'+str(int(coeff[1]*1000)/1000)+'*e+'+str(int(coeff[2]*1000)/1000))
# plt.title('')
# plt.xlabel('Epsilon')    
# plt.ylabel('Success probability')
# plt.legend(loc = 'lower right')
# plt.show()

