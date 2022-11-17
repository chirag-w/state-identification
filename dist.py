import numpy as np
from util import *
from sdp import *
import matplotlib
from matplotlib import pyplot as plt 

zero_state = np.array([[1,0],[0,0]], dtype = complex)
plus_state = np.array([[0.5,0.5],[0.5,0.5]], dtype = complex)
one_state = np.array([[0,0],[0,1]], dtype = complex)

prob_sdp = []
prob_pgm = []
weight = []
for p in np.arange(0,1.05,0.05):
    mixed_state = p*zero_state+(1-p)*one_state
    try:
        p_sdp,_ = state_identification([mixed_state,plus_state])
        prob_sdp.append(p_sdp)
        p_pgm,_ = pretty_good_measurement([mixed_state,plus_state])
        prob_pgm.append(p_pgm)
        weight.append(p)
    except:
        print('Optimization did not succeed for p =',p)

plt.plot(weight, prob_sdp, label = 'SDP Probability')
plt.plot(weight, prob_pgm, label = 'PGM Probability')
plt.title('Distinguishing p|0><0|+(1-p)|1><1| and |+><+|')
plt.xlabel('p')
plt.ylabel('Success probability')
plt.legend(loc = 'lower right')
plt.show()