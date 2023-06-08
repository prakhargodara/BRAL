import numpy as np
import itertools
import matplotlib.pyplot as plt
from os.path import exists

varrybois = np.arange (1,11,dtype = int)

iterations = 5000
periods = np.arange (1,101)


dirname = "Single_sims/rand_noisy_pl/"
A_avg = np.zeros ((len(varrybois)))
xipm_avg = np.zeros ((len(varrybois)))
xith_avg = np.zeros ((len(varrybois)))
Var_avg = np.zeros ((len(varrybois)))





for j in range (len(varrybois)):
		bets = np.zeros ((iterations,4,100))
		xips = np.zeros ((iterations,3,100))
		xims = np.zeros ((iterations,3,100))
		for k in range (1,iterations+1):
			loaded = np.load (dirname+'rand_sim_{0}_{1}.npy'.format(int(varrybois[j]),k))
			
			bets[k-1] = loaded[0]
			xips[k-1] = loaded[1][1:,:]
			xims[k-1] = loaded[2][1:,:]
			
		dists = (xips[:,:,:-1] - xips[:,:,1:])**2 + (xims[:,:,:-1] - xims[:,:,1:])**2
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

		xipm_avg[j] = len(inds2)/(iterations*4*100)#np.average (dists)#r_xi
		xith_avg[j] = np.average (dists)
		
		
		
		#plt.plot (np.average (np.average (bets[:,1:,:],axis=0),axis=0))
		#plt.ylim(-1,21)

		A_avg[j] = np.average (bets[:,1:,:])	
		Var_avg[j] = np.average(Var_per_riter_avg)#/A_avg[j,i]



## FIG 7................
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2.0
from matplotlib.ticker import FormatStrFormatter




plt.rcParams["figure.figsize"] = [10.375, 8.375]
plt.rcParams["figure.autolayout"] = True
fig, ax_new = plt.subplots()
fig.subplots_adjust(right=0.75)
left, bottom, width, height = [.52, 0.17, 0.35, 0.35]
ax = fig.add_axes([left, bottom, width, height])



twin1 = ax.twinx()

# Offset the right spine of twin2.  The ticks and label have already been
# placed on the right by twinx above.

p1, = ax.plot(varrybois,xith_avg, "k--", label=r"$\langle l \rangle$")
p2, = twin1.plot(varrybois,xipm_avg, "k:", label="$f$")#1.6*(np.ones(len(pl_nums)))  #0.4*(np.arange(4,len(pl_nums)+4)

twin1.tick_params(axis='y', direction = 'in',length = 7,width = 2)
ax.tick_params(axis='y', direction = 'in',length = 7,width = 2)
ax.tick_params(axis='x', direction = 'in',length = 7,width = 2)

ax.set_ylim(0.03,0.06)
ax.set_xlabel("$\sigma$",fontsize=18)
twin1.set_ylim(0.25,0.45)

ax.set_yticks (np.arange(0.03,0.075,0.015),fontsize = 18)
ax.set_xticks(np.arange(0,15,5),fontsize = 18)
ax.set_xticklabels(np.arange(0,15,5,dtype=int),fontsize = 18)
ax.set_yticklabels(np.arange(0.03,0.075,0.015),fontsize = 18)


twin1.set_yticks (np.arange(0.25,0.54,.1),fontsize = 18)
twin1.set_yticklabels (np.arange(0.25,0.54,.1),fontsize = 18)
twin1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))


ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())

tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)

ax.legend(handles=[p1, p2],fontsize=18)






ax_new.plot (varrybois, A_avg,'k-')

ax_new.tick_params(axis='y', direction = 'in',length = 7,width = 2)
ax_new.tick_params(axis='x', direction = 'in',length = 7,width = 2)


ax_new.set_yticks (np.arange(0,12,2,dtype=int),fontsize = 18)
ax_new.set_xticks(np.arange(0,12,2),fontsize = 18)
ax_new.set_xticklabels(np.arange(0,12,2,dtype=int),fontsize = 18)
ax_new.set_yticklabels(np.arange(0,12,2,dtype=int),fontsize = 18)
#ax_new.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax_new.set_xlabel ('$\sigma$',fontsize=18)
ax_new.set_ylim(0,10)
ax_new.set_ylabel (r'$\langle A \rangle$',fontsize=18)
plt.tight_layout()
plt.show()










