import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

##sim5 K=2,sim11 K=1.9 (Varying priors)


xip = np.arange (0.,2.5,0.5)
xim = np.arange (0.,2.5,0.5)
iterations = 1000

print (xip)

A_avg = np.zeros ((len(xip),len(xim)))
Var_avg = np.zeros ((len(xip),len(xim)))

xipm_20 = np.zeros (A_avg.shape)

for i in range (len(xim)):
	for j in range (len(xip)):
		All_bets = np.zeros ((iterations,4,100))
		All_xipms = np.zeros ((iterations,4,100))
		for k in range (1,iterations+1):
			#if (j== 10 or j==11 or j==12 or j==13 or j== 9 or j==16 or j==15 or j==14):
			
			simname = "sim_{0}_{1}_{2}.npy".format (round (xip[j]*10), round(xim[i]*10),k)
			
			#else:
				#simname = "sim_{0}_{1}_{2}.npy".format (int (xip[i]*10), int(xim[j]*10),k)
			dirname = "Single_sims/Final_sims/xi_sims/"
			
			AA = np.load (dirname+simname)
			All_bets[k-1] = AA[0]
			All_xipms[k-1] =  AA[1]
			
		av_xipms = np.average (np.average (All_xipms,axis=0),axis=0)
		av_bets = np.average (np.average (All_bets,axis=0),axis=0)
		
		#plt.plot (av_xipms.T)
		#plt.ylim (-1.2,1.2)
		#plt.plot (i,np.average (av_bets[80:]),'o')	
			
		## Some quantities that are calculated
		
		xipm_20[j,i] = np.average (All_xipms[:,:,90])
		Var_per_riter_avg = np.average (np.std (All_bets,axis=1),axis=0)
		
		
		
		Avg_bets_allp = np.average (All_bets,axis=0)
		Avg_bets = np.average (Avg_bets_allp,axis=0)
		
		##Convergence
		#AB_AV = np.average (np.average (All_bets,axis=1),axis=1)
		#print (xim[i],xip[j])
		#plt.plot (np.cumsum(AB_AV)/np.arange (1,iterations+1))
		#plt.show()
		
		
		##Bets stats
		#print (xip[i])
		#for rounds in range (19):
			#plt.hist (All_bets[:,:,5*rounds].flatten(),density = True)
			#plt.show()
		
		A_avg[j,i] = np.average (All_bets)	
		Var_avg[j,i] = np.average(Var_per_riter_avg)/A_avg[j,i]
		
plt.show()

plt.imshow (xipm_20,origin = 'lower',extent = [-1,1,-1,1.],aspect = 1)	
plt.colorbar()
plt.show()
'''
'''
plt.plot (Var_avg,A_avg, 'ko')
plt.ylabel (r"$\langle A \rangle$")
plt.xlabel (r"$\langle Var \rangle$")	
plt.show()

for j in range (len(xim)):
	
	plt.plot (xip,A_avg[j],'-x',label = "LL = {}".format(xim[j]))
plt.legend()
plt.ylim(0,20)
plt.xlabel ("$K$")
plt.ylabel (r"$\langle A \rangle$")	
plt.show()

for j in range (len(xip)):
	
	plt.plot (xim,A_avg[:,j],'-x',label = "K = {}".format(xip[j]))
plt.legend()
plt.ylim(0,20)
plt.xlabel ("$LL$")
plt.ylabel (r"$\langle A \rangle$")	
plt.show()

for j in range (len(xim)):
	
	plt.plot (xip,Var_avg[j],'-x',label = "LL = {}".format(xim[j]))
plt.legend()
plt.xlabel ("$K$")
plt.ylabel (r"$\langle Var \rangle$")	
plt.show()

for j in range (len(xip)):
	
	plt.plot (xim,Var_avg[:,j],'-x',label = "K = {}".format(xip[j]))
plt.legend()
plt.xlabel ("$LL$")
plt.ylabel (r"$\langle Var \rangle$")	
plt.show()








### Interpolating the functions

nx = len (xip)
ny = len (xim)
act_x,act_y = np.meshgrid (xip,xim)
points = np.column_stack ((act_y.ravel(),act_x.ravel()))
values = np.zeros (nx*ny)
values2 = np.zeros (nx*ny)

for i in range (len(values)):
	values[i] = A_avg[np.where (xim == points[i,0])[0], np.where (xip == points[i,1])[0]]
	values2[i] = Var_avg[np.where (xim == points[i,0])[0], np.where (xip == points[i,1])[0]]
	



print (points)

y = np.linspace (xip[0],xip[-1],100)
x = np.linspace (xim[0],xim[-1],100)

want_x,want_y = np.meshgrid (x,y)
#want_points = np.column_stack ((want_y.ravel(),want_x.ravel()))	
	
gridA = griddata (points,values, (want_x,want_y),method = 'cubic')
gridVar = griddata (points,values2, (want_x,want_y),method = 'cubic')

plt.imshow(gridA.T,extent = [-1,1,-1,1.],aspect = 1,origin = 'lower')
plt.colorbar()
plt.show()

plt.imshow(gridVar.T,extent = [-1.,1,-1,1.],aspect = 1,origin = 'lower')
plt.colorbar()
plt.show()

plt.plot (y,gridVar[-1,:])
plt.plot (y,gridVar[-20,:])
plt.plot (y,gridVar[-40,:])
plt.plot (y,gridVar[-60,:])
plt.plot (y,gridVar[-80,:])
plt.plot (y,gridVar[0,:])
plt.show()




plt.plot (y,gridA[-1,:])
plt.plot (y,gridA[-20,:])
plt.plot (y,gridA[-40,:])
plt.plot (y,gridA[-60,:])
plt.plot (y,gridA[-80,:])
plt.plot (y,gridA[0,:])
plt.show()

print (gridA)
