import sys
import time
import os
from reduction import *
import numpy as np
import scipy.optimize as opt
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import random
from multiprocessing import Pool
import itertools
from centrality_measures import distance_centrality,d_i, adjacent_groups,min_dist_group,adj_mat,group_degree
from g_matrix_generator import *
import networkx as nx
import netgraph
from os.path import exists
import pyinform as pi
from scipy import stats

group_IDs = np.arange (grid_params.N_groups)
player_IDs = np.arange (grid_params.N_players)

K=3
gammap=9
nnum = 1
subdir = 'big_nets/'



randm = 32
#dirname = "Old_grid_data/Arbit_grid_data/gamma_p/Network_big_1/K=3/m=10/".format(K,gammap,nnum,randm)
dirname = "Rand_grids/64_25/4_3_20/"
GR_matrix = np.load(dirname+'arbit_grm_K{0}0_LL{1}0.npy'.format (K,gammap))


#Plotting the network#

NP = int(np.max(GR_matrix))+1
NG = len(GR_matrix[:,0])

pl_per_gr = d_i(GR_matrix,NP)		
				

group_centrality = group_degree(GR_matrix)
dc = distance_centrality (GR_matrix)
min_dist = min_dist_group(GR_matrix)


'''  Now we plot the trajectories'''

iterations = 1500

bets = []
benes = []

A_total = []

for i in range (iterations):
	if  exists(dirname+'arbit_grid_sim_K{0}0_LL{2}0_{1}.npy'.format(K,i+1,gammap)):
		bets .append(np.load (dirname+'arbit_grid_sim_K{0}0_LL{2}0_{1}.npy'.format(K,i+1,gammap)))
		#benes.append(np.load(dirname+'benev_pl_indices_{}.npy'.format(i+1)))
	

		
bets = np.array (bets)


print (bets.shape)

grp_A = np.average (np.average(np.average(bets,axis=2),axis=0),axis=1)
var_A = np.average (np.std(np.average(bets,axis=2),axis=2),axis=0)
plt.plot (group_centrality, grp_A,'o')
plt.show()

gc = np.unique (group_centrality)

grA = []
for i in range (len(gc)):
	inds = np.where (group_centrality == gc[i])[0]
	grA.append(np.average (grp_A[inds]))
	plt.plot (gc[i],grA[-1],'ko' )
plt.show()

grA = np.array (grA)

print ()
savedir = 'universality_graphs/'

#print (dirname[11:16],dirname[17:23])  #dirname[11:17],dirname[18:24]
np.save (savedir + 'centrality_{0}_{1}'.format (dirname[11:16],dirname[17:23] ),gc)
np.save (savedir+'grpA_{}_{}'.format (dirname[11:16],dirname[17:23] ),grA)

'''
g1 = []
g2 = []
g3 = []
g4 = []
g5 = []

for i in range (len(bets)):
	
	if (benes[i][0] == 1):
		g1.append(i)
	if (benes[i][0] == 4):
		g2.append(i)
	if (benes[i][0] == 7):
		g3.append(i)
	if (benes[i][0] == 10):
		g4.append(i)
	if (benes[i][0] == 13):
		g5.append(i)
				
G = []
G.append(g1)
G.append(g2)
G.append(g3)
G.append(g4)
G.append(g5)

norm_bets = np.load("K3_bets.npy")
norm_av = np.average (np.average (np.average (norm_bets,axis=0),axis=1),axis=1)
for i in range (len(G)):
	it_av_bets = np.average (bets[G[i]],axis=0)
	av_bets = np.average (np.average (it_av_bets,axis=1),axis=1)
	plt.plot (av_bets,'ko')
	
	plt.plot (norm_av,'ro')
	plt.show()
	
	
	
## distance vs mutual information plots

mdg = 	min_dist_group(GR_matrix)

for i in range (len(G)):
	gr_av_bets = np.average (bets[G[i]],axis=2)
	MI = np.zeros ((NG,NG))
	Corr = np.zeros ((NG,NG))
	
	for j in range (NG):
		for k in range (j,NG):
			MI[j,k] =pi.mutual_info (gr_av_bets[:,j,:].flatten() ,gr_av_bets[:,k,:].flatten() )
			MI[k,j] = MI[j,k]
			
			Corr[j,k] = stats.pearsonr(gr_av_bets[:,j,:].flatten() ,gr_av_bets[:,k,:].flatten())[0]
			
			Corr[k,j] = Corr [j,k]
			
			
	#plt.imshow(1./MI,origin = 'lower')
	#plt.colorbar()
	#plt.show()
	
	#plt.imshow(1./Corr,origin = 'lower')
	#plt.colorbar()
	#plt.show()
	
	#plt.plot (mdg[i],MI[i],'o')
	#plt.show()
	
	plt.plot (mdg[i],Corr[i],'o')
	plt.ylim (0,1)
	plt.show()

	
'''


