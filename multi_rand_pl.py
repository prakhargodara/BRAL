import numpy as np
import itertools
import matplotlib.pyplot as plt
from os.path import exists

pl_nums = np.arange (4,16,dtype=int)
#pl_nums=np.delete(pl_nums,3)
iterations = 1000

periods = np.arange (1,101)


dirname = "Single_sims/Final_sims/rand_multi_pls_cons_alpha/"

#_varyalpha
#_10r
bins = np.linspace (0,20,10)
var_avg = []
A_avg = []
xipm_avg = []
xi_th_avg = []


for i in pl_nums:
	#B = np.zeros (len(bins)-1)
	all_bets = []
	xips = []
	xims = []
	its = 0
	for j in range(1,iterations+1):
		#print (j,i)
		if  exists(dirname+'pl_{0}_rand_sim_{1}.npy'.format(i,j)):
			its+=1
		
			loaded = np.load(dirname+'pl_{0}_rand_sim_{1}.npy'.format(i,j))
			all_bets.append( loaded[0])
			xips.append( loaded[1])
			xims.append( loaded[2])
		else: 
			continue	
		#b,e = np.histogram (np.average (all_bets[j-1],axis=1), bins =bins )
		#plt.plot (e[1:],b)
		#plt.show()
		#B =B+b 
	#b,e =np.histogram(np.average (all_bets,axis=2).flatten(),density = True,bins=5)
	#B/=iterations
	#B/=i
	#plt.plot (e[1:],B,'-x')
	#plt.ylim (0,0.2)
	#plt.show()
	
	#plt.show()
	print (its)
	all_bets = np.array (all_bets)
	print (all_bets.shape)
	xips = np.array (xips)
	xims = np.array (xims)
	rxi = np.sqrt((xips**2) + (xims**2))
	
	plt.plot(np.average (np.average (all_bets,axis=0),axis=0))
	
	
	
	
	yerr = np.average (np.std (all_bets,axis=1),axis=0)
	var_avg .append( np.std (np.average (np.average (all_bets,axis=1),axis=1)))	
	A_avg .append( np.average (all_bets))
	#xipm_avg.append( (np.mean ( (xips**2) + (xims**2) )))
	xipm_avg.append(len(np.where (rxi.flatten()<0.1)[0])/len(rxi.flatten()) )
	xi_th_avg.append( np.mean(np.arctan (xips,xims  )))
	#plt.errorbar (periods, np.average (np.average (all_bets,axis=0),axis=0).T,yerr = yerr)
	#if (i==6 or i==20 or i == 4):
		#plt.plot (periods, np.average (np.average (all_bets,axis=0),axis=0).T)
	
	#plt.plot (yerr)
	#plt.plot ( np.average (np.average (xips,axis=0),axis=0).T)
	#plt.plot ( np.average (np.average (xims,axis=0),axis=0).T)
	#plt.ylim (-3,23)
	#plt.show()
	#plt.plot (np.average (np.average (all_bets,axis=0),axis=0))

plt.show()





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

p1, = ax.plot(pl_nums, xipm_avg, "k--", label=r"$P(r_\xi<0.1)$")
p2, = twin1.plot(pl_nums,0.4*(np.ones(len(pl_nums))), "k:", label=r"$\alpha$")#1.6*(np.ones(len(pl_nums)))  #0.4*(np.arange(4,len(pl_nums)+4)

ax.set_ylim(0,0.24)
ax.set_xlabel(" $N$",fontsize=18)
#ax.set_ylabel(r"$P(r_\xi<0.1)$",fontsize=18)
twin1.set_ylabel(r"$\alpha$",fontsize=18)
twin1.set_ylim(0,1)


twin1.tick_params(axis='y', direction = 'in',length = 7,width = 2)
ax.tick_params(axis='y', direction = 'in',length = 7,width = 2)
ax.tick_params(axis='x', direction = 'in',length = 7,width = 2)

ax.set_yticks (np.arange(0,0.30,0.06),fontsize = 18)
ax.set_xticks(np.arange(4,16,2),fontsize = 18)
ax.set_xticklabels(np.arange(4,16,2,dtype=int),fontsize = 18)
ax.set_yticklabels(np.arange(0,0.30,0.06),fontsize = 18)

twin1.set_yticks (np.arange(0,1.5,0.5),fontsize = 18)
twin1.set_yticklabels (np.arange(0,1.5,0.5),fontsize = 18)
twin1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))





tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)

ax.legend(handles=[p1, p2],fontsize=18)






ax_new.errorbar (pl_nums, A_avg,yerr=var_avg,capsize=9,color='k')
ax_new.set_xlabel (' $N$',fontsize=18)
ax_new.set_ylim(0,13.3)
ax_new.set_ylabel (r'$\langle A \rangle$',fontsize=18)

ax_new.tick_params(axis='y', direction = 'in',length = 7,width = 2)
ax_new.tick_params(axis='x', direction = 'in',length = 7,width = 2)

ax_new.set_yticks (np.arange(0,16,4,dtype=int),fontsize = 18)
ax_new.set_xticks(np.arange(4,16,2),fontsize = 18)
ax_new.set_xticklabels(np.arange(4,16,2,dtype=int),fontsize = 18)
ax_new.set_yticklabels(np.arange(0,16,4,dtype=int),fontsize = 18)
#ax_new.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.show()






plt.errorbar (pl_nums, A_avg,yerr = var_avg)
plt.xlabel ('Group size $N$')
plt.ylabel (r'$\langle A \rangle$')
plt.show()

plt.plot (pl_nums,var_avg)
plt.show()



## Running this code with more iterations, higher K, \xipm search space should be finer


'''
Multi_pls varyalpha has K,gp = 2,0.9 and 2,0.
Multi_pls2 has K,gp = 2,0.9 and 3,0.9 and 2,0.
Multi_pls_10r has K,gp = 2,0.9
'''
