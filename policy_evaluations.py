import numpy as np
from reduction import *
import matplotlib.pyplot as plt
import time


def wideness (dist):
	return -kldiv (dist,np.ones (21)/21.)




tau = 20
N = 4
xip = 0.1
xim = 0.5
var = 3
m=10
n_steps = 50
K=3.

prior = discretegaussian(tau, m, 5)




#transition = statetransitions_paper(tau, N, xim, xip, var)

#policy = bratpolicy_new(prior, transition, nrofsteps=n_steps, compcost=K)


Xip = np.linspace (-1.5,1.5,10)
Xim = np.linspace (-1.5,1.5,10)
rxi = np.sqrt (Xip**2 + Xim**2)
thxi = np.arctan2 (Xip,Xim)

plt.plot (rxi,thxi)
plt.show()

MAT = np.zeros ((len(Xip), len(Xim), 21,61  ))

tt = time.time()

for i in range (len(Xip)):
	for j in range (len(Xim)):
		transition = statetransitions_paper(tau, N, Xim[j], Xip[i], var)
		policy = bratpolicy_new(prior, transition, nrofsteps=n_steps, compcost=K)
		
		for k in range (21):
			for l in range (61):
				MAT[i,j,k,l] = wideness(policy[0,k,l])
				
				
