import numpy as np
import pulp as p

def density_matrix(pure_states):
    #Convert an array of pure states to their density matrix representation
    N = len(pure_states)
    S = []
    for i in range(N):
        S.append(np.outer(pure_states[i],pure_states[i].conjugate()))
    return S

def tensor_power(S,n):
    assert(n >= 1)
    rho = []
    N = len(S)
    for i in range(N):
        rho.append(S[i].copy())
        for j in range(n-1):
            rho[i] = np.kron(rho[i],S[i])
    return rho

def trace_distance(rho1,rho2):
    eig = np.abs(np.linalg.eigvalsh(rho1-rho2))
    return sum(eig)/2

def linear_upper_bound(y,x):
    # Determine a,b such that y >= ax+b for all y,x 
    # that minimizes absolute error
    assert(len(y) == len(x))
    LUB = p.LpProblem('Linear upper bound', p.LpMinimize)
    
    a = p.LpVariable("a")
    b = p.LpVariable("b")

    sum_y = sum(y)
    sum_x = sum(x)
    LUB += a*sum_x+b-sum_y
    for i in range(len(y)):
        LUB += a*x[i]+b >= y[i]
        LUB += a*x[i]+b >= 0.0
    #print(LUB)

    status = LUB.solve()
    #print("Status:" ,p.LpStatus[status])

    print(p.value(a),p.value(b))

    return p.value(a),p.value(b)