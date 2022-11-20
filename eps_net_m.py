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
def gen_states(N):
    theta = 2*np.pi/N
    R = RX(theta)
    cur_state = np.array([1,0],dtype = complex)
    S = []
    for i in range(N):
        S.append(cur_state)
        cur_state = R@cur_state
    return density_matrix(S)

def gen_two_qubit_states(N):
    states = gen_states(N)
    S = []
    for i in range(N):
        for j in range(N):
            S.append(np.kron(states[i],states[j]))
    return S

def gen_m_qubit_states(N,m):
    num_states = N**m
    states = gen_states(N)
    S = []
    for i in range(num_states):
        index = i
        curr_state = states[index%N] 
        for j in range(m-1):
            index = int(index/N)
            curr_state = np.kron(states[index%N], curr_state)
        S.append(curr_state)
    return S

N_min = 2
N_max = 6
n_min = 1
n_max = 1

prob_pgm = np.zeros((N_max+1,n_max+1),dtype = float)
prob_info = np.zeros((N_max+1,n_max+1),dtype = float)
eps = np.zeros(N_max+1,dtype = float)

p_info = []
for N in range(N_min,N_max+1):
    m = 3
    S = gen_m_qubit_states(N,m)
    tr_dist = trace_distance(S[0],S[1])
    for i in range(len(S)):
        for j in range(i):
            tr_dist = min(tr_dist,trace_distance(S[i],S[j]))
    eps[N] = tr_dist
    print(N," states")
    print("epsilon = ",tr_dist)
    for n in range(n_min,n_max+1):
        print(n," copies")
        p_pgm,_ = pretty_good_measurement(S,n)
        print("PGM success probability: ",p_pgm)
        p_info = info_theory_upper_bound(S,n)
        print("Information-theoretic upper bound on success probability: ",p_info)
        try:
            p_opt,_ = state_identification(S,n)
            print("Optimal success probability: ",p_opt)
        except:
            print("Could not optimize for n = ",n)
        prob_pgm[N,n] = p_pgm
        prob_info[N,n] = p_info

f1 = plt.figure()
for N in range(N_min,N_max+1):
    plt.plot(np.arange(n_min,n_max+1),prob_pgm[N,n_min:n_max+1], label = 'N = '+str(N)+', eps = '+str(eps[N]))
plt.title('')
plt.xlabel('Number of copies')    
plt.ylabel('Success probability')
plt.legend(loc = 'lower right')
plt.show()


f2 = plt.figure()
for N in range(N_min,N_max+1):
    plt.plot(np.arange(n_min,n_max+1),prob_info[N,n_min:n_max+1]/prob_pgm[N,n_min:n_max+1], label = 'N = '+str(N))
plt.title('Tightness of upper bound')
plt.xlabel('Number of copies')    
plt.ylabel('UB/LB')
plt.legend(loc = 'upper right')
plt.show()


col = ['blue','orange','green','red','black']
for n in range(n_min,n_max+1):
    f = plt.figure()
    plt.plot(eps[N_min:N_max+1],prob_pgm[N_min:N_max+1,n], label = 'n = '+str(n), color = col[n%5])
    coeff = quad_upper_bound(prob_pgm[N_min:N_max+1,n],eps[N_min:N_max+1])
    x_temp = np.linspace(eps[N_max],1,num = 20)
    print(x_temp)
    plt.plot(x_temp,coeff[0]*(x_temp**2)+coeff[1]*x_temp+coeff[2],linestyle = 'dashed', color = col[n%5], label = 'p = '+str(int(coeff[2]*1000)/1000)+'*e^2+'+str(int(coeff[1]*1000)/1000)+'*e+'+str(int(coeff[2]*1000)/1000))
    plt.title('')
    plt.xlabel('Epsilon')    
    plt.ylabel('Success probability')
    plt.legend(loc = 'lower right')
    plt.show()

