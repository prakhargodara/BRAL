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
from centrality_measures import distance_centrality,  adjacent_groups,min_dist_group
from g_matrix_generator import *
from os.path import exists
import pyinform as pi
from scipy import stats
group_IDs = np.arange (grid_params.N_groups)
player_IDs = np.arange (grid_params.N_players)

K=3
gammap=9
nnum = 4
subdir = 'big_nets/'

randm =1
dirname = "Arbit_grid_data/gamma_p/Network_big_1/New_gamma/".format(K,gammap,nnum,randm)
simnaame = 'arbit_grm_K0_LL90.npy'
GR_matrix = np.load(dirname+'arbit_grm_K{0}0_LL{1}0.npy'.format (K,gammap))

print (GR_matrix)
#Plotting the network#

NP = int(np.max(GR_matrix))+1
NG = len(GR_matrix[:,0])

pl_per_gr = np.zeros (NP)
for i in range (NP):
	pl_per_gr[i] = len (np.where (GR_matrix==i)[0])




def gc_order (order):
	adj_mat = np.zeros ((NG,NG))
	
	for i in range (NG):
		for j in range (NG):
			group_i = GR_matrix [i]
			group_j = GR_matrix [j]
			
			adj_mat[i,j] = len(np.intersect1d (group_i, group_j))
			adj_mat [i,i] = 0
	
	
	prev_order = np.sum (adj_mat,axis=0)
	#prev_order = prev_order#/np.max(prev_order)
	next_order = np.zeros (NG)
	for j in range (2,order+1):
		for i in range (NG):
			next_order[i] = np.sum (prev_order[np.where (adj_mat [i]!=0)[0]])
		
		#next_order = next_order#/np.max(next_order)	
		prev_order = next_order
		
	return prev_order
		
				

gc = gc_order(1)
dc = distance_centrality(GR_matrix)
group_centrality_2 = gc_order(2)

'''  Now we plot the trajectories'''

iterations = 200

bets = []
benes = []

A_total = []
P_CG = []
G_CG = []
D_CG = []

dc = distance_centrality (GR_matrix)

for i in range (iterations):
	if  exists(dirname+'arbit_grid_sim_K{0}0_LL{2}0_{1}.npy'.format(K,i+1,gammap)):
		bets .append(np.load (dirname+'arbit_grid_sim_K{0}0_LL{2}0_{1}.npy'.format(K,i+1,gammap)))
	'''
	benes.append(np.load(dirname+'benev_pl_indices_{}.npy'.format(i+1)))
	
	gcgs = 0
	lengs = 0
	dcgs = 0
	for j in range (randm):
		grps = np.where (GR_matrix == benes[-1][j])[0]
		lengs += len (grps)
		gcgs += np.sum (group_centrality[grps])
		dcgs+= np.sum (dc[grps])
		
	P_CG.append (lengs)
	G_CG.append (gcgs)
	D_CG.append (dcgs)
'''	
bets = np.array (bets)
print (bets.shape)
#np.save ("K3_bets",bets)

#bets = bets[:,:,:,20:80]

av_bets = np.average(bets,axis=2)

A_group = np.average (np.average (av_bets,axis=2),axis=0)

plt.plot (gc,A_group,'o')

plt.show()


av_bet2 = np.zeros ((av_bets.shape[1], av_bets.shape[0]*av_bets.shape[2] ))

for i in range (av_bets.shape[1]):
	av_bet2[i] = av_bets[:,i,:].flatten()
	
av_bets = av_bet2



MI = np.zeros ((NG,NG))
Corr = np.zeros ((NG,NG))
for i in range (NG):
	for j in range (i,NG):
		MI[i,j] = pi.mutual_info (av_bets[i],av_bets[j])
		MI[j,i] = MI[i,j]
	
		Corr[i,j] = stats.pearsonr(av_bets[i],av_bets[j])[0]
		#print (stats.pearsonr(av_bets[i],av_bets[j]))
		Corr[j,i] = Corr[i,j]	
mdg = 	min_dist_group(GR_matrix)

plt.imshow(mdg)
plt.colorbar()
plt.show()

plt.imshow(1./Corr)
plt.colorbar()
plt.show()


plt.imshow(1./MI)
plt.colorbar()
plt.show()



for i in range (NG):
	#plt.plot (mdg[i],MI[i],'o')
	plt.plot (mdg[i],Corr[i],'o')
	plt.ylim (0.,1.1)
	plt.show()
		






'''

P_CG = np.array (P_CG)
G_CG = np.array (G_CG)
benes = np.array (benes)
bet10 = []
bet3 = []
betlen = []
avg_K = []

int_gr = []
for i in range (iterations):
	for j in range (len (benes[i])):
		inds = np.where (GR_matrix == benes[i,j])
		grps = inds [0]
		poss = inds [1]
		bet10.append  (np.average (bets [i,grps,poss]))
		
		bet3.append ((np.sum (bets [i,grps])- np.sum (bets [i,grps,poss]))/(3.*len(grps)*100) )
		if (bet10[-1] >= 5):
			int_gr.append (grps)
		betlen.append ( len(grps) )
		
int_gr = list(itertools.chain.from_iterable(int_gr))
		
bet10 = np.array(bet10)
bet3 = np.array(bet3)	
betlen = np.array (betlen)
avg_K = np.array (avg_K)

int_gr = np.array (int_gr)

print (np.average (bet10),np.average (bet3))

for i in range (NG):
	plt.plot (group_centrality[i],len (np.where (int_gr==i)[0]),'ko')
plt.show()



for i in range (1,5):
	inds = np.where (betlen==i)[0]
	#plt.hist (bet10[inds])
	plt.plot (i,np.average(bet10[inds]),'o')
plt.show()


print (np.average (bet10))
'''
#print (np.average (bets[:,adjacent_groups(14,GR_matrix)[0]]),adjacent_groups(14,GR_matrix))

#plt.hist(np.average (np.average (np.average (bets,axis=1),axis=1),axis=1))
#plt.show()

#wtv_bets = np.average (np.average (np.average (bets,axis=1),axis=1),axis=1)
#plt.plot (P_CG,np.average (np.average (np.average (bets,axis=1),axis=1),axis=1),'o')
#plt.show()
#plt.plot (G_CG,np.average (np.average (np.average (bets,axis=1),axis=1),axis=1),'o')
#plt.show()

#plt.plot (D_CG,np.average (np.average (np.average (bets,axis=1),axis=1),axis=1),'o')
#plt.show()


'''
nx = 10
H,xe,ye = np.histogram2d (G_CG,wtv_bets,bins = (nx,20),density = True)
norm =  np.histogram (G_CG,bins = nx,density = True)[0]

print (len(ye))

for i in range (len(ye)-1):
	H[:,i] = H[:,i]/norm


print (H)
plt.hist2d (P_CG,wtv_bets)
plt.colorbar()
plt.show()
plt.hist2d (G_CG,wtv_bets)
plt.colorbar()
plt.show()
plt.imshow(H.T,extent =(2,65,6.3,10.3),aspect = 50/4)
plt.colorbar()
plt.show()


bets = np.average (bets,axis=0)
plt.plot (np.average (bets,axis = 1).T)
plt.show()

'''

avg_A = np.average(np.average (bets,axis=2),axis=0)

plt.plot (gc,avg_A,'o')
#plt.ylim (5,13)
plt.xlabel ("$C_g$")
plt.ylabel (r"$\langle A \rangle_{group}$")
#plt.show()

plt.plot (dc,avg_A,'o')
#plt.ylim (5,13)
plt.xlabel ("$C_g$")
plt.ylabel (r"$\langle A \rangle_{group}$")
#plt.show()

