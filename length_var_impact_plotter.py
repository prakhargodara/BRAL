import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


#Length impact plotter

'''

dirname = 'Length_impact/'
LEs = np.load (dirname+'LEs.npy')
Ls = np.array ([10,25,50,75,100,200])

for i in Ls:
	A_avg = np.load (dirname+'A_{}.npy'.format(i))
	
	plt.plot (LEs,A_avg[:,2],label = 'L={}'.format(i))
	
plt.legend()
plt.show()
'''

#Var impact plotter

dirname = 'Var_impact/'

simnames = ['_001','','_99']
Les = np.array ([0.01,0.5,0.99])
for i in range (len(simnames)):
	A = np.load (dirname+'A_avg'+simnames[i]+'.npy')
	
	plt.plot (A[0].T,'-o',label = '$K=3$, $\gamma_p = {}$'.format(Les[i]))
	
	plt.plot (A[1].T,'-x',label = '$K=3.5$, $\gamma_p = {}$'.format(Les[i]))
	
plt.legend()
plt.ylim(0,15)
plt.xlabel ('$\sigma$')
plt.ylabel (r'$\langle A \rangle$')
plt.show()
