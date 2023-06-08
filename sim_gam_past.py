import sys
import time

from numba import float64, int64
from numba import types
from multiprocessing import Pool
from mean_rms import*
from reduction import *
from opt_xipm_gam_past import *
import numpy as np
import scipy.optimize as opt
from scipy.optimize import fsolve


def discretegaussian(mean, variance):
    dist = np.exp(-(np.arange(21) - mean) ** 2 / (2 * (variance) ** 2))
    dist /= dist.sum()
    return dist 



def simulation_nolearn (xip,xim,K,gamma,priors,initial_bets,gamel):
	params.game_length = gamel
	N = len (xip)
	bets = np.zeros ((N,params.game_length ),dtype=int)
	bets [:,0] = initial_bets
	
	policies = []
	
	for i in range (N):
		#print (i)
		transitions = statetransitions_paper(params.tokens, N, xim[i], xip[i], params.var_transition)
		policies.append (bratpolicy_new (priors[i], transitions, nrofsteps = params.game_length-1,nrofplayers=N,gamma = gamma[i],compcost = K[i] ))
		#print (policies[0].shape)
		
	for i in range (1,params.game_length):
		for j in range (N):
			others = np.delete (np.arange (0,N), j)
			others_bets = np.sum (bets [others], axis=0)[i-1]
			prevact = bets[j,i-1]
			
			probabilities = policies[j][i-1,prevact,others_bets]
			bets[j,i] = np.random.choice (np.arange (params.tokens+1),1,p=probabilities)
		
	return bets
	
	
def simulation_learn (xip,xim,K,gamma,gamma_p,priors,initial_bets,gamel):
	np.random.seed()
	params.game_length = gamel
	N = len (xip)
	bets = np.zeros ((N,params.game_length ),dtype=int)
	xips_return = np.zeros ((N,params.game_length ))
	xims_return = np.zeros ((N,params.game_length ))
	
	bets [:,0] = initial_bets
	
	policies = []
	
	for i in range (N):
		#print (i)
		transitions = statetransitions_paper(params.tokens, N, xim[i], xip[i], params.var_transition)
		policies.append (bratpolicy_new (priors[i], transitions, nrofsteps = params.game_length-1,nrofplayers=N,gamma = gamma[i],compcost = K[i] ))



		
	for i in range (1,params.game_length):
		period = i-1
		for j in range (N):
		
			if (i==1):
				others = np.delete (np.arange (0,N), j)
				others_bets = np.sum (bets [others], axis=0)[i-1]
				prevact = bets[j,i-1]
				#np.random.seed (int(time.time()+i*j))
				probabilities = policies[j][period,prevact,others_bets]
				bets[j,i] = np.random.choice (np.arange (params.tokens+1),1,p=probabilities)
				xips_return[j,i] = xip[j]
				xims_return[j,i] = xim[j]
			
			else: 
				learning_period = bets[:,0:i]
					
				xip[j],xim[j] = optimal_xipm_solo (learning_period,xip[j],xim[j],gamma_p[j],j)
				transitions = statetransitions_paper(params.tokens, N, xim[j], xip[j], params.var_transition)
				policies[j] = (bratpolicy_new (priors[j], transitions, nrofsteps = params.game_length-i,nrofplayers=N,gamma = gamma[j],compcost = K[j] ))
				period = 0
				
				others = np.delete (np.arange (0,N), j)
				others_bets = np.sum (bets [others], axis=0)[i-1]
				prevact = bets[j,i-1]
				#np.random.seed (int(time.time()+i*j))
				probabilities = policies[j][period,prevact,others_bets]
				bets[j,i] = np.random.choice (np.arange (params.tokens+1),1,p=probabilities)
				xips_return[j,i] = xip[j]
				xims_return[j,i] = xim[j]
				
				

		
	return bets,xips_return,xims_return
		








''' 

A = 0.1*np.ones (4)
B = 0.5*np.ones (4)
E = 10*np.ones (4,dtype = int)
E[0] = 0
E[1] = 20

D = 0.9*np.ones (4)
C = 10.*np.ones (4)


M = np.ones ((4,21))/21.


plt.plot(simulation_nolearn (A,B,C,D,M,E).T)
plt.show()

 
   
for i in range (4):
	M[i] = discretegaussian (10.,5)

LL = 7*np.ones (4,dtype = int)

#bets = 10*np.ones ((4,5),dtype = int)

#bets [0] = np.zeros(5,dtype = int)

#bets [1] = 20*np.ones(5,dtype = int)

#print (bets)

#print (optimal_xipm_solo(bets,0.1,0.5,1))

#plt.plot (simulation_learn (A,B,C,D,M,LL,E).T)
#plt.show()



def multi (i):
	AA = simulation_learn (A,B,C,D,M,LL,E).T
	return AA

iterations = 16
p=Pool(16)

tt = time.time()

AA =p.map(multi, range(iterations))

AA = np.array (AA)

print ((time.time()-tt)/60., "mins")

plt.plot (np.average (AA,axis=0))
plt.show()

plt.plot (np.average (AA,axis=1))
plt.show()


#print (AA[1][3],AA[0][3])
	
	
	

'''



