import numpy as np
import itertools
import matplotlib.pyplot as plt
from os.path import exists

varrybois = np.arange (1,11,dtype = int)

iterations = 2000
Ks = np.array ([3.,3.5,4,4.5,5])

periods = np.arange (1,101)


dirname = "Single_sims/one_noisy_pl_001/"
A_avg = np.zeros ((len(Ks),len(varrybois)))
xipm_avg = np.zeros ((len(Ks),len(varrybois)))
xith_avg = np.zeros ((len(Ks),len(varrybois)))
Var_avg = np.zeros ((len(Ks),len(varrybois)))




for i in range (len(Ks)):
	XIMMY = []
	XIPPY = []
	for j in range (len(varrybois)):
		bets = np.zeros ((iterations,4,100))
		xips = np.zeros ((iterations,3,100))
		xims = np.zeros ((iterations,3,100))
		for k in range (1,iterations+1):
			loaded = np.load (dirname+'sim_{0}_{1}_{2}.npy'.format(round (Ks[i]*10),int(varrybois[j]),k))
			
			bets[k-1] = loaded[0]
			xips[k-1] = loaded[1][1:,:]
			xims[k-1] = loaded[2][1:,:]
			
		dists = (xips[:,:,:-1] - xips[:,:,1:])**2 + (xims[:,:,:-1] - xims[:,:,1:])**2
		XIMMY.append(xims[:,:,-1])
		XIPPY.append(xips[:,:,-1])
		inds = np.where (xips.flatten() > 0)[0]
		inds2 = np.where (xims.flatten()[inds] > 0)[0]
			
		## Some quantities that are calculated


		Var_per_riter_avg = np.average (np.std (bets,axis=1),axis=0)
		A_grp = np.average (bets,axis=2)
		#Var_per_riter_avg = np.average (np.std(A_grp,axis=1))
		acfs = np.zeros (21)

		Avg_bets_allp = np.average (bets,axis=0)
		Avg_bets = np.average (Avg_bets_allp,axis=0)
		

		
		r_xi = (xips**2 + xims**2)#[:,:,20:80]
		th_xi  = np.arctan2 (xips,xims)#[:,:,20:80]

		xipm_avg[i,j] = len(inds2)/(iterations*4*100)#np.average (dists)#r_xi
		xith_avg[i,j] = np.average (dists)
		
		
		
		#plt.plot (np.average (np.average (bets[:,1:,:],axis=0),axis=0))
		#plt.ylim(-1,21)

		A_avg[i,j] = np.average (bets[:,1:,:])	
		Var_avg[i,j] = np.average(Var_per_riter_avg)#/A_avg[j,i]
	#plt.show()
	
	XIP = np.array (XIPPY).flatten()
	XIM = np.array (XIMMY).flatten()
	
	#plt.hist2d (XIP,XIM,bins = 20,range = [[-1.5, 1.5], [-1.5, 1.5]])
	#plt.show()

plt.plot (varrybois,A_avg[0], '-go',label = r'$K=3$')
plt.plot (varrybois,A_avg[1], '-ro',label = r'$K=3.5$')
plt.plot (varrybois,A_avg[2], '-ko',label = r'$K=4$')
plt.plot (varrybois,A_avg[3], '-bo',label = r'$K=4.5$')
plt.plot (varrybois,A_avg[4], '-yo',label = r'$K=5$')
plt.ylabel (r"$\langle A\rangle$")#
plt.xlabel (r"$\sigma$")
plt.ylim(0,20.0)

np.save ('A_avg',A_avg)
	
plt.legend()

plt.show()














