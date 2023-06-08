import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy.ndimage
#import statsmodels.api as sm
##sim5 K=2,sim11 K=1.9 (Varying priors)




Ks = np.load ('K_vals_all2.npy')
LEs = np.load ('gp_vals_all2.npy')


Ks2 = np.load ('K_vals10.npy')
LEs2 = np.load ('gp_vals10.npy')

Ks3 = np.load ('K_vals11.npy')
LEs3 = np.load ('gp_vals11.npy')
'''

#print (len(Ks2))





np.save ('K_vals_all2',np.concatenate ((Ks,np.concatenate ((Ks2,Ks3)))))
np.save ('gp_vals_all2',np.concatenate ((LEs,np.concatenate ((LEs2,LEs3)))))
'''
Ks4 = np.concatenate ((Ks2,Ks3))#np.load ('K_vals_all.npy')
LEs4 = np.concatenate ((LEs2,LEs3))#np.load ('gp_vals_all.npy')


grps = 2		##Max = 758560
print (Ks.shape[0]/4)
points  = np.vstack((Ks,LEs))
values = np.zeros (len(Ks))
A = np.zeros (len(Ks))

#plt.hist (LEs)
#plt.show()




grp_A = np.zeros (grps)
grp_K = np.zeros ((grps,4))
grp_LE = np.zeros ((grps,4))
grp_var = np.zeros (grps)

bets = []
xips = []
xims = []



for i in range (1,grps+1):
	grp_Ks = Ks [4*(i-1):4*i]
	grp_LEs = LEs [4*(i-1):4*i]
	
	load = np.load ('Single_sims/Final_sims/rand_sims/rand_sim_{}.npy'.format(i))
	grp_bets = load[0]
	
	bets.append (grp_bets)
	xips.append (load[1])
	xims.append (load[2])
	
	A_t = np.sum (grp_bets,axis=0)
	
	Gains = np.zeros (grp_bets.shape)
	for j in range (4):
		Gains[j] =  0.4*A_t - grp_bets[j] 
		
	Total_gains = np.average (Gains,axis=1)
	
	values [4*(i-1):4*i] = Total_gains 
	A [4*(i-1):4*i] = np.average (grp_bets,axis=1)
	
	grp_K[i-1] = grp_Ks
	grp_LE[i-1] = grp_LEs #- np.min (grp_LEs)
	grp_A[i-1] = np.average (grp_bets)
	grp_var[i-1] = np.average (np.std (grp_bets,axis=0))

bets = np.array (bets)
xips = np.array (xips)
xims = np.array (xims)	
grp_xip = np.average (np.average (xips,axis=1),axis=1)
grp_xim = np.average (np.average (xims,axis=1),axis=1)

rxi = xips**2 + xims**2
thxi = np.arctan2(xips,xims)

grp_rxi = np.average (np.average (rxi,axis=1),axis=1)
grp_txi = np.average (np.average (thxi,axis=1),axis=1)

#plt.scatter (rxi.flatten(),bets.flatten())
#plt.show()


#plt.hist2d (thxi.flatten(),bets.flatten(),bins=20)
#plt.colorbar()
#plt.show()

	
def P_condi_gains (K,gamma_p,Ks,LEs,values):
	inds = np.where (np.abs(Ks-K)< 0.1)[0]
	values2 = values[inds]
	Ks2 = Ks[inds]
	LEs2 = LEs[inds]
	inds = np.where (np.abs(LEs2-gamma_p)< 0.025)[0]
	values2 = values2[inds]
	
	return values2
'''

inds = np.where ((grp_K.flatten() - 4.7)**2 <= 0.01)
	
glef = grp_LE.flatten()[inds]
gabf = np.average (bets,axis=2).flatten()[inds]

kess = np.linspace (0.,1.,15)
dl = kess[1] - kess[0]
for i in range (len(kess)):
	inds = np.where ((glef-kess[i])**2 <= (dl/2.)**2 )[0]
	plt.plot (kess[i], np.average (gabf[inds]),'ko')
plt.ylim(0,20)



plt.show()
	
	
	
'''	



co_K = np.arange (0.1,5.1,0.2)
co_G = np.arange(0.025,1.025,0.2/5.)
condi = np.zeros ((len(co_K),len(co_G)))
nums = np.zeros ((len(co_K),len(co_G)))
for i in range (len(co_K)):
    for j in range (len(co_G)):
        #vals,eds = np.histogram(P_condi_gains (co_K[i],co_G[j],Ks,LEs,values),bins = bins,density = True )
        #d_eds = eds[1]-eds[0]
        
        #avg_vals = np.sum (vals*eds[1:]*d_eds + vals*eds[:-1]*d_eds)/2
        zakky = P_condi_gains (co_K[i],co_G[j],Ks,LEs,values)
        
        condi[i,j] = np.average (zakky)#avg_vals
        nums[i,j] = len (zakky)
    #if (i == 19 or i== len(co_K)-1):
        #plt.plot (co_G,condi[i])
#plt.show()

#np.save ('condi1',condi)
#np.save ('nums1',nums)

c1 = np.load('condi1.npy')
#c2 = np.load('condi2.npy')

n1 = np.load('nums1.npy')
#n2 = np.load('nums2.npy')

#condi = np.multiply(n1,c1) + np.multiply(n2,c2)

condi = c1#condi/(n1+n2)


plt.rcParams["figure.figsize"] = [10.375, 8.375]
plt.rcParams["figure.autolayout"] = True
plt.imshow(condi.T,origin = 'lower',extent = [0,5,0,1],aspect = 5,interpolation = 'sinc',cmap = 'Greys')

plt.xlabel ("$K$",fontsize = 18)
plt.ylabel (r"$ \gamma_p $",fontsize = 18)
plt.tick_params('both',direction = 'in',length = 7,width = 2)
plt.xticks (np.arange(0,6,dtype=int),labels = np.arange(0,6,dtype=int),fontsize = 18)
plt.yticks (np.arange(0,1.2,0.2),labels = np.round(np.arange(0,1.2,0.2),1),fontsize = 18)
#plt.xticklabels (Ks,fontsize = 18)
#plt.yticklabels (np.arange(0,1.2,0.2),fontsize = 18)
plt.tight_layout()	

cbar = plt.colorbar()
cbar.set_label( r'$\langle G | K,\gamma_p \rangle$',fontsize=18)
cbar.set_ticks(np.arange(3,6),fontsize = 18)
cbar.set_ticklabels(np.arange(3,6),fontsize = 18)
condi = scipy.ndimage.zoom(condi,2)

plt.contour(condi.T,extent = (0,5,0,1),colors = ['#5a5c5a'],levels = [3,3.5,4,4.5,4.8],linewidths=3)
#plt.savefig ('gains_condi.png')
plt.show()	
	
	

plt.show()	
	

'''	
inds = np.where (grp_A>14)[0]
H,_,_ = np.histogram2d(grp_K[inds].flatten(),grp_LE[inds].flatten(),density = True,bins=10)
plt.imshow(H.T,origin = 'lower',extent = [0,5,0,1],aspect = 5,interpolation = 'bilinear')
plt.show()



ma = values.max()
mi = values.min()
plt.hist (values,density = True,bins = 100,alpha = 0.5)
#inds = np.where ((LEs-0.7)**2<0.01)[0]

plt.plot (ma,0,'x')
plt.plot (mi,0,'x')
plt.xlabel ('Average gain $G$')
plt.ylabel ('$P(G)$')
plt.show()
#plt.hist (values[inds],density = True,bins = 100,alpha = 0.5)
#plt.show()

#plt.scatter (grp_K,grp_LE,c=grp_var)
#plt.show()





## How do gains depend on params?


di = ma-mi
its =5
for i in range (its):
	ranje = mi + np.array ([1./its*i,1./its*(i+1)])*di
	
	int_len = ranje[1]-ranje[0]
	mid_val  = np.average (ranje)
	
	inds = np.where ((values-mid_val)**2 <= int_len**2/4)[0]
	print ('percentage of players',len(inds)/len(Ks))
	
	H,_,_ = np.histogram2d(Ks[inds],LEs[inds],density = True,bins=20)
	plt.imshow(H.T,origin = 'lower',extent = [0,5,0,1],aspect = 5,interpolation = 'bilinear')
	plt.colorbar()
	plt.xlabel ('$K$')
	plt.ylabel ('$\gamma_p$')
	plt.title ('Gains in {0}-{1}'.format(ranje[0],ranje[1]))
	plt.show()
	

	
	
## How do contributions depend on params?

ma = A.max()
mi = A.min()
di = ma-mi

plt.hist (A,density = True,bins = 100,alpha = 0.5)
#inds = np.where ((LEs-0.7)**2<0.01)[0]

plt.plot (ma,0,'x')
plt.plot (mi,0,'x')
plt.xlabel (r' $\langle A \rangle$')
plt.ylabel (r'$P(\langle A \rangle)$')
plt.show()


for i in range (its):
	ranje = mi + np.array ([1./its*i,1./its*(i+1)])*di
	
	int_len = ranje[1]-ranje[0]
	mid_val  = np.average (ranje)
	
	inds = np.where ((A-mid_val)**2 <= int_len**2/4)[0]
	
	H,_,_ = np.histogram2d(Ks[inds],LEs[inds],density = True,bins=30)
	plt.imshow(H.T,origin = 'lower',extent = [0,5,0,1],aspect = 5,interpolation = 'bilinear')
	plt.colorbar()
	plt.xlabel ('$K$')
	plt.ylabel ('\gamma_p')
	plt.title ('Contributions')
	plt.title ('Contributions in {0}-{1}'.format(ranje[0],ranje[1]))
	plt.show()
	
	
'''	
##Gains and Contris
'''


inds = np.where (abs(LEs-.950) <=0.05)[0]#abs(Ks-3.3) <=0.8
plt.hist (A[inds],alpha = 0.5,density = True)
plt.hist (values[inds],alpha = 0.5,density = True)
plt.xlim (-15,20)
plt.show()

inds = np.where (abs(LEs-.550) <=0.05)[0]#abs(Ks-3.3) <=0.8
plt.hist (A[inds],alpha = 0.5,density = True)
plt.hist (values[inds],alpha = 0.5,density = True)
plt.xlim (-15,20)
plt.show()

#plt.scatter (A[inds],values[inds])
#plt.show()

H,_,_ = np.histogram2d(A[inds],values[inds],density = True)
plt.imshow(H.T,origin = 'lower',extent = [A.min(),A.max(),values.min(),values.max()],interpolation = 'bilinear')
plt.colorbar()
plt.xlabel ('$ A $')
plt.ylabel ('$G$')
plt.title (r'$\gamma_p \in [0.7,1.]$')
plt.show()




######################################################################

##conditional cooperation tests##########





x = np.linspace (-20,20.100)


aa = []
bb = []

for i in range (grps):
	grbets2 = bets[i].copy()
	grbets = grbets2.copy()
	
	avgrbets = np.average (grbets,axis=0)
	
	for kkk in range (4):
		grbets[kkk] = grbets2[kkk] - avgrbets
		
	aa.append(grbets[:,:-1])
	bb.append(bets[i,:,1:] - bets[i,:,:-1])	
	#plt.plot (grbets[:,:-1], bets[i,:,1:] - bets[i,:,:-1],'ko' )

	#plt.ylim (-0.1,20.1)
	#plt.show()
print (bets.shape)
a = np.array (aa).flatten()
b = np.array (bb).flatten()

print (a.shape)
H,_,_ = np.histogram2d (a,b,bins=20,density = True)


plt.imshow(H.T,origin = 'lower',extent = [a.min(),a.max(),b.min(),b.max()])
plt.colorbar()
plt.xlabel ('Diff of my bets and average bets in round i')
plt.ylabel ('Increase of my bets from round i to i+1')
plt.show()


plt.contour (H.T)
plt.xlabel ('Diff of my bets and average bets in round i')
plt.ylabel ('Increase of my bets from round i to i+1')
plt.show()



cc = []
dd = []

def conditional_cooperation_condi_K (K,Ks):
	inds = np.where (np.abs(Ks-K)< 0.1)[0]
	K_groups = inds//4
	K_grp_pos = np.remainder (inds,4)
	
	
	for i in range (len(K_groups)):
		grp = K_groups[i]
		cc.append (aa[i][K_grp_pos[i]])
		dd.append (bb[i][K_grp_pos[i]])
	
	c = np.array (cc).flatten()
	d = np.array (dd).flatten()
	return c,d
	
c,d = 	conditional_cooperation_condi_K (0.1,Ks)
print (c.shape)
H,_,_ = np.histogram2d (c,d,bins=20,density = True)


plt.imshow(H.T,origin = 'lower',extent = [a.min(),a.max(),b.min(),b.max()])
plt.colorbar()
plt.xlabel ('Diff of my bets and average bets in round i')
plt.ylabel ('Increase of my bets from round i to i+1')
plt.show()


plt.contour (H.T)
plt.xlabel ('Diff of my bets and average bets in round i')
plt.ylabel ('Increase of my bets from round i to i+1')
plt.show()

'''
