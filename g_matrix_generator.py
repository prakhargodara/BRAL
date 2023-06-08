import numpy as np
import matplotlib.pyplot as plt


class grid_params:

	N_players = 64
	
	N_groups = 25
	
	N_pl_per_gr = 4
	
	game_length = 100


''' Below we make a network'''

group_IDs = np.arange (grid_params.N_groups)
player_IDs = np.arange (grid_params.N_players)


'''GR_matrix has player nums for each group. Its a random matrix with consistency condition satisfied.

GR_matrix has the shape
[[1 2 3 4]
 [5 6 7 8]
 [4 6 2 8]]
 
So group 1 has 4 players (1,2,3,4), group 2 has (5,6,7,8) and group 3 has (4,6,2,8).
'''

D_GRM = 100*np.ones (grid_params.N_groups*grid_params.N_pl_per_gr)
D_GRM [0:grid_params.N_players] = player_IDs		##Consistency
np.random.shuffle(D_GRM)

GR_matrix = D_GRM.reshape ((grid_params.N_groups, grid_params.N_pl_per_gr))
for i in range (grid_params.N_groups):
	gm = GR_matrix [i]
	all_spots = np.arange (4)
	open_spots = np.where (gm==100)[0]
	occupied_spots = np.delete (all_spots,open_spots)
	
	if  open_spots.size == 0:
		continue
	else:
		players = np.array (gm[occupied_spots],dtype = int)
		allowed_players = np.delete (player_IDs,players)
		new_ps = np.random.choice (allowed_players,len(open_spots),replace = False)
		k=0
		for j in range (4):
			if (GR_matrix[i,j]==100):
				GR_matrix[i,j] = new_ps[k]
				k+=1












## Ring generation.

def ring (N):
	Np = 3*N
	GRM = np.zeros ((N,4))
	GRM [0] = np.array ([0,1,2,3])
	
	for i in range (1,N):
		GRM[i,0] = GRM[i-1,-1]
		for k in range (1,4):
			GRM [i,k] = GRM [i,k-1]+1
			
	GRM [N-1,-1] = GRM [0,0]

	return GRM
	
	
def linear (N):
	Np = 3*N+1
	GRM = np.zeros ((N,4))
	GRM [0] = np.array ([0,1,2,3])
	
	for i in range (1,N):
		GRM[i,0] = GRM[i-1,-1]
		for k in range (1,4):
			GRM [i,k] = GRM [i,k-1]+1
			
	return GRM	


NG = grid_params.N_groups
def gc_order (order,GR_matrix):
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


def rand_mat ():

	D_GRM = 100*np.ones (grid_params.N_groups*grid_params.N_pl_per_gr)
	D_GRM [0:grid_params.N_players] = player_IDs		##Consistency
	np.random.shuffle(D_GRM)

	GR_matrix = D_GRM.reshape ((grid_params.N_groups, grid_params.N_pl_per_gr))
	for i in range (grid_params.N_groups):
		gm = GR_matrix [i]
		all_spots = np.arange (4)
		open_spots = np.where (gm==100)[0]
		occupied_spots = np.delete (all_spots,open_spots)
	
		if  open_spots.size == 0:
			continue
		else:
			players = np.array (gm[occupied_spots],dtype = int)
			allowed_players = np.delete (player_IDs,players)
			new_ps = np.random.choice (allowed_players,len(open_spots),replace = False)
			k=0
			for j in range (4):
				if (GR_matrix[i,j]==100):
					GR_matrix[i,j] = new_ps[k]
					k+=1
					
	return GR_matrix

#print (linear(5))
	
#a = rand_mat()
#print (a)
#np.save ('test_mat',linear(5))	
'''

##Find the node centrality statistics..

iterations = 1000

all_cgs = np.zeros ((iterations, NG))

for i in range (iterations):
	all_cgs[i] = gc_order(1,rand_mat())
	

nums = 15

histogram = np.zeros (nums)

for i in range (nums):
	histogram[i] = len(np.where (all_cgs.flatten()==i)[0])
	
plt.plot (histogram/np.sum(histogram), '-x')
plt.show()

'''
#print (len(ring(10)[:,0]))	

#print (ring(5))
	
#np.save ('ring_GRM',ring(5))	
	
	
#np.save('gr_matrix',GR_matrix)
