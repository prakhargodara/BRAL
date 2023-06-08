import sys
import time

from reduction import *
from opt_xipm_gam_past import *
import numpy as np
import scipy.optimize as opt
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import random
from multiprocessing import Pool
import itertools

from g_matrix_generator import *

## Player numbers (ID)


## Group numbers (ID) 


## For each group ID, 4 player ID's. (Each player at least one group)




def discretegaussian(mean, variance):
    dist = np.exp(-(np.arange(21) - mean) ** 2 / (2 * (variance) ** 2))
    dist /= dist.sum()
    return dist 	
	

def simulation_learn (xip,xim,K,gamma,priors,gamma_p,initial_bets):

	
	bets = np.zeros ((grid_params.N_groups,grid_params.N_pl_per_gr,grid_params.game_length ),dtype=int)
	bets [:,:,0] = initial_bets
	
	policies = []
	
	N = grid_params.N_players
	
	for i in range (N):
		#print (i)
		transitions = statetransitions_paper(params.tokens, grid_params.N_pl_per_gr, xim[i], xip[i], params.var_transition)
		policies.append (bratpolicy_new (priors[i], transitions, nrofsteps = grid_params.game_length-1,gamma = gamma[i],compcost = K[i] ))



		
	for i in range (1,grid_params.game_length):
		period = i-1
		for k in range (N):
			A = np.where (GR_matrix==k)
			member_groups = A[0]
			index_in_group = A[1]
			
			
			if (i==1) or (gamma_p[k]==0):
				for j in range (len (member_groups)):
					others = np.delete (np.arange (0,4), index_in_group[j])
					others_bets = np.sum (bets [member_groups[j],others,i-1])
					prevact = bets[member_groups[j],index_in_group[j],i-1]
					np.random.seed (int(time.time()+i*j))
					probabilities = policies[k][period,prevact,others_bets]
					bets[member_groups[j],index_in_group[j],i] = np.random.choice (np.arange (params.tokens+1),1,p=probabilities)
			
			
			else: 
				
				learning_period = bets[member_groups,:,0:i]
				print (member_groups,k,i,index_in_group)

				xip[k],xim[k] = optimal_xipm_solo_grid (learning_period,xip[k],xim[k],gamma_p[k],index_in_group)
				#print (xip[j],xim[j],'XIPM vals')
				transitions = statetransitions_paper(params.tokens, 4, xim[k], xip[k], params.var_transition)
				
				policies[k] = (bratpolicy_new (priors[k], transitions, nrofsteps = grid_params.game_length-i,gamma = gamma[k],compcost = K[k] ))
				period = 0
			
			
				
				for j in range (len (member_groups)):
					others = np.delete (np.arange (0,4), index_in_group[j])
					others_bets = np.sum (bets [member_groups[j],others,i-1])
					prevact = bets[member_groups[j],index_in_group[j],i-1]
				
					probabilities = policies[k][period,prevact,others_bets]
					np.random.seed (int(time.time()+i*j))
					bets[member_groups[j],index_in_group[j],i] = np.random.choice (np.arange (params.tokens+1),1,p=probabilities)
		
	return bets








''' Running a simulation! '''


''' Below we make a network'''

group_IDs = np.arange (grid_params.N_groups)
player_IDs = np.arange (grid_params.N_players)

GR_matrix = np.load('test_mat.npy')
#Plotting the network#

NP = grid_params.N_players
NG = grid_params.N_groups


group_locations = np.zeros ((NG,2))


radius = 10
theta = 2.*np.pi/NG
for i in range (NG):
	angle = i*theta
	group_locations[i] = [radius* np.cos(angle), radius* np.sin(angle)]
	
	
	plt.plot (group_locations[i][0],group_locations[i][1], 'o')
for j in range (NP):
	AA = np.where (GR_matrix==j)[0]	#Group_nums
	
	if len(AA) == 1:
		continue
	else:
		num_mem_g = len(AA)
		all_points = np.zeros ((2,num_mem_g))
		for mmm in range (num_mem_g):
			all_points [:,mmm] = group_locations[AA[mmm]]
		plt.plot( *zip(*itertools.chain.from_iterable(itertools.combinations(all_points.T,2))))
		
plt.show()

'''  Now we want the PGG to be played'''



xip = 0.1*np.ones (NP)
xim = 0.5*np.ones (NP)
K = 3.*np.ones (NP)
gamma = 0.9*np.ones (NP)
learnlengths = 0.9*np.ones (NP)

m = 10.*np.ones (NP)
priors = np.zeros ((NP,params.tokens+1))
for i in range (NP):
	priors [i] = discretegaussian (m[i],5)

initial_bets =  10*np.ones ((NG,4),dtype=int)

initial_bets [1] = np.array ([4,16,2,20])

plt.plot (np.average (simulation_learn (xip,xim,K,gamma,priors,learnlengths,initial_bets),axis=1).T )
plt.show()


