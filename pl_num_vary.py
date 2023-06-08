import numpy as np
import itertools
import matplotlib.pyplot as plt
##This file plots from Multi_pls2, vary alpha and 10r

pl_nums = np.arange (4,16,dtype=int)
#pl_nums=np.delete(pl_nums,3)
K = 2.
gp = 0.9		#This can be 0 or 0.9
iterations = 240

periods = np.arange (1,101)


dirname = "Single_sims/Final_sims/Multi_pls2/"

#_varyalpha
#_10r
bins = np.linspace (0,20,10)
var_avg = []
A_avg = []
xipm_avg = []
xi_th_avg = []
for i in pl_nums:
	#B = np.zeros (len(bins)-1)
	all_bets = np.zeros ((iterations,i,len(periods)))
	xips = np.zeros ((iterations,i,len(periods)))
	xims = np.zeros ((iterations,i,len(periods)))
	for j in range(1,iterations+1):
		all_bets[j-1] = np.load(dirname+'pl_{0}_num_sim_{1}_{2}_{3}.npy'.format(i,round (K*10),round(gp*100),j))[0]
		xips[j-1] = np.load(dirname+'pl_{0}_num_sim_{1}_{2}_{3}.npy'.format(i,round (K*10),round(gp*100),j))[1]
		xims[j-1] = np.load(dirname+'pl_{0}_num_sim_{1}_{2}_{3}.npy'.format(i,round (K*10),round(gp*100),j))[2]
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
	yerr = np.average (np.std (all_bets,axis=1),axis=0)
	var_avg .append( np.std (np.average (np.average (all_bets,axis=1),axis=1)))	
	A_avg .append( np.average (all_bets))
	xipm_avg.append( (np.mean ( (xips**2) + (xims**2) )))
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









plt.rcParams["figure.figsize"] = [10.375, 8.375]
plt.rcParams["figure.autolayout"] = True
fig, ax_new = plt.subplots()
fig.subplots_adjust(right=0.75)
left, bottom, width, height = [.60, 0.50, 0.25, 0.25]
ax = fig.add_axes([left, bottom, width, height])



twin1 = ax.twinx()

# Offset the right spine of twin2.  The ticks and label have already been
# placed on the right by twinx above.

p1, = ax.plot(pl_nums, xipm_avg, "b-", label=r"$\langle \xi_+^2 + \xi_-^2 \rangle$")
p2, = twin1.plot(pl_nums,1.6*(np.ones(len(pl_nums))), "r-", label="MPCR")


ax.set_xlabel("Group size $N$")
ax.set_ylabel(r"$\langle \xi_+^2 + \xi_-^2 \rangle$")
twin1.set_ylabel("MPCR")

ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())

tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)

ax.legend(handles=[p1, p2])






ax_new.errorbar (pl_nums, A_avg,yerr=var_avg)
ax_new.set_xlabel ('Group size $N$')
#ax_new.set_ylim(2,11)
ax_new.set_ylabel (r'$\langle A \rangle$')
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
