import numpy as np
from scipy.optimize import LinearConstraint,minimize

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

def quad_upper_bound(y,x):
    # Determine coefficients of y = ax^2+bx+c
    # that minimize squared error
    assert(len(y) == len(x))
    def loss(coeff):
        coeff = np.asarray(coeff)
        error = 0.0
        for i in range(len(x)):
            error += (coeff[0]*x[i]*x[i]+coeff[1]*x[i]+coeff[2]-y[i])**2
        return error
    constr = LinearConstraint([[x[i]**2,x[i],1] for i in range(len(x))],np.zeros(len(x)),[np.inf for i in range(len(x))])
    x0 = np.array([0,0,0])
    res = minimize(loss,x0,constraints=[constr])
    print(res)
    return res.x

def von_neumann_entropy(rho):
    p = np.linalg.eigvalsh(rho)
    S = 0.
    for i in range(len(p)):
        if(p[i]<=0):
            continue
        S-= p[i]*np.log2(p[i])
    return S

def bin_entropy(eta):
    if(eta == 0):
        return 0
    return eta*np.log2(1/eta)+(1-eta)*np.log2(1/(1-eta))

def bin_search(f, params, val, low, high, error):
    #Binary search for x in [low,high] such that val-error<f(x,params)<val+error
    #For a monotonic increasing function f 
    x = (low+high)/2
    while abs(f(x,params)-val) >= error:
        if f(x,params) > val:
            high = x
        elif f(x,params) < val:
            low = x
        else:
            break
        x = (low+high)/2
    return x

def holevo_info(S):
    N = len(S)
    mixed_state = sum(S)/N
    Chi = von_neumann_entropy(mixed_state)
    for i in range(N):
        Chi -= von_neumann_entropy(S[i])/N
    return Chi

def f(x,y):
    return x
