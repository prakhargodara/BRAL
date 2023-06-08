import sys
import time

from numba import float64, int64
from numba import types

from reduction import *
import numpy as np
import scipy.optimize as opt
from scipy.optimize import fsolve



class params:
	var_transition = 3.
	tokens = 20
	game_length = 20
	alpha = 0.4
	search_grid_size = 61

	

	
def likelihood_xip ( xip,gamma_p, bets, plno, nrofplayers,prev_xip):
	
	
	T = len (bets[0])
	trans_prob = statetransitions_paper (params.tokens, nrofplayers, 0., xip, params.var_transition)
	others = np.delete (np.arange (0,nrofplayers), plno)
	others_bets = np.sum (bets [others], axis=0)
	
	k=0.
	for j in range (T-1):
		if (trans_prob [bets[plno,j], others_bets[j], others_bets[j+1] ] < 10**-7):
			k+=-8
		else:
			
			k = k + np.log( trans_prob [bets[plno,j], others_bets[j], others_bets[j+1] ])*(gamma_p**(T-2-j))
		
	return gamma_p*k - (1-gamma_p)*(prev_xip-xip)**2
	
	

	
	
def likelihood_xim ( xim,gamma_p, bets, plno, nrofplayers,prev_xim):
	
	
	T = len (bets[0])
	
	
	trans_prob = statetransitions_paper (params.tokens, nrofplayers, xim, 0., params.var_transition)
	others = np.delete (np.arange (0,nrofplayers), plno)
	others_bets = np.sum (bets [others], axis=0)
	
	k=0.
	for j in range (T-1):
		if (trans_prob [bets[plno,j], others_bets[j], others_bets[j+1] ] < 10**-7):
			k+=-8
		else:
			
			k = k + np.log( trans_prob [bets[plno,j], others_bets[j], others_bets[j+1] ])*(gamma_p**(T-2-j))
		
	return gamma_p*k - (1-gamma_p)*(prev_xim-xim)**2




def optimal_xipm_solo (bets,prev_xip,prev_xim,gamma_p,plno):
	#bets has to have size NxT (N = number of players, T = number of rounds)
	#print (bets,prev_xip,prev_xim,plno)
	N = len(bets[:,0])
	T = len (bets[0])
	xips = prev_xip
	xims = prev_xim
	
	XIP = np.linspace (-1.5,1.5,params.search_grid_size)
	XIM = np.linspace (-1.5,1.5,params.search_grid_size)
	
	vfunc1 = np.vectorize (likelihood_xip ,excluded =['gamma_p','bets','plno','nrofplayers','prev_xip'])
	
	vfunc2 = np.vectorize (likelihood_xim ,excluded =['gamma_p','bets','plno','nrofplayers','prev_xim'])
	
	others = np.delete (np.arange (0,N), plno)
	others_bets = np.sum (bets [others], axis=0)[:-1]
	agent_bets = bets [plno][:-1]
	
	A = others_bets - agent_bets * (N-1)
	B = np.where (A>0)[0]
	C = np.where (A==0)[0]
	D = np.where (A<0)[0]
	if (len(C) == len (A)):
		return prev_xip,prev_xim
	else:
		if (len (B)>0):
			LIK = vfunc2 (xim = XIM,gamma_p=gamma_p, bets = bets, plno = plno, nrofplayers = N , prev_xim = xims)
			
			xims  = XIM [np.argmax (LIK)]
			
		if (len (D)>0):
			LIK = vfunc1 (xip = XIP,gamma_p=gamma_p, bets = bets, plno = plno, nrofplayers = N, prev_xip = xips )
			xips  = XIP [np.argmax (LIK)]
			
			
	return (xips,xims)



	
	
'''  Stuff for grids'''	

def likelihood_xim_grid ( xim, bets,gamma_p, group_indices, nrofplayers):
	
	N_groups = len (group_indices)
	T = len (bets[0,0])
	
	
	trans_prob = statetransitions_paper (params.tokens, nrofplayers, xim, 0., params.var_transition)
	
	
	others = []		##Assuming 4 players only
	others_bets = []
	agent_bets = []
	for j in range (N_groups):
		
		others . append (np.delete (np.arange (0,4), group_indices[j]))
		others_bets. append ( np.sum (bets [j,others[j]], axis=0))
		agent_bets . append ( bets [j,group_indices[j]])
		
	others_bets = np.array (others_bets)
	agent_bets = np.array (agent_bets)
	
	k=0.
	for ng in range (N_groups):
		for j in range (T-1):
			k = k + np.log( trans_prob [bets[ng,group_indices[ng],j], others_bets[ng][j], others_bets[ng][j+1] ]) *(gamma_p**(T-2-j))
		
	return k	
	
	
	
def likelihood_xip_grid ( xip, bets,gamma_p, group_indices, nrofplayers):
	
	N_groups = len (group_indices)
	T = len (bets[0,0])
	
	
	trans_prob = statetransitions_paper (params.tokens, nrofplayers, 0, xip, params.var_transition)
	
	
	others = []		##Assuming 4 players only
	others_bets = []
	agent_bets = []
	for j in range (N_groups):
		
		others . append (np.delete (np.arange (0,4), group_indices[j]))
		others_bets. append ( np.sum (bets [j,others[j]], axis=0))
		agent_bets . append ( bets [j,group_indices[j]])
		
	others_bets = np.array (others_bets)
	agent_bets = np.array (agent_bets)
	
	k=0.
	for ng in range (N_groups):
		for j in range (T-1):
			k = k + np.log( trans_prob [bets[ng,group_indices[ng],j], others_bets[ng][j], others_bets[ng][j+1] ]) *(gamma_p**(T-2-j))
		
	return k


	
def optimal_xipm_solo_grid (learning_period,xip,xim,gamma_p,index_in_group):
	N_groups = len (index_in_group)
	
	if (N_groups == 1):
		return optimal_xipm_solo (learning_period[0],xip,xim,gamma_p,index_in_group[0])
	else:
		xips = xip
		xims = xim
	
		XIP = np.linspace (-2,2,params.search_grid_size)
		XIM = np.linspace (-2,2,params.search_grid_size)
	
		vfunc1 = np.vectorize (likelihood_xip_grid ,excluded =['bets','gamma_p','group_indices','nrofplayers'])
	
		vfunc2 = np.vectorize (likelihood_xim_grid ,excluded =['bets','gamma_p','group_indices','nrofplayers'])
		
		
		others = []		##Assuming 4 players only
		others_bets = []
		agent_bets = []
		for j in range (N_groups):
		
			others . append (np.delete (np.arange (0,4), index_in_group[j]))
			others_bets. append ( np.sum (learning_period [j,others[j]], axis=0)[:-1] )
			agent_bets . append ( learning_period [j,index_in_group[j]][:-1] )
		
		

		others_bets = np.array (others_bets).flatten()
		agent_bets = np.array (agent_bets).flatten()
		
		
		A = others_bets - agent_bets * (3)
		B = np.where (A>0)[0]
		C = np.where (A==0)[0]
		D = np.where (A<0)[0]
		if (len(C) == len (A)):
			return (xips,xims)
		else:
			if (len (B)>0):
				LIK = vfunc2 (xim = XIM, bets = learning_period,gamma_p = gamma_p, group_indices = index_in_group, nrofplayers = 4 )
			
				xims  = XIM [np.argmax (LIK)]
			
			if (len (D)>0):
				LIK = vfunc1 (xip = XIP,bets = learning_period,gamma_p = gamma_p, group_indices = index_in_group, nrofplayers = 4 )
				xips  = XIP [np.argmax (LIK)]
			
		
	return (xips,xims)
	
	
	
	
