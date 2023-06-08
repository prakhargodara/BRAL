import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

##Given a 10x4 matrix we calculate the mean and the rms contribution vectors in this code




def mean_rms_f(x,gamel):

	if (x.shape[0] == gamel):
		mean = np.average(x, axis = 1)
		diff = x-mean[:,None]
		
		rms = np.sqrt(np.average(diff**2,axis=1))
	
		return  mean, rms
		
	elif (x.shape[0]!=4):
		return mean_contri(x.T)
		
	else:	
		print ("This is not what the function does")
		return None
		
		
		
def rms_asym(x,gamel):

	if (x.shape[0] == gamel):
		mean = np.average(x, axis = 1)
		diff = x-mean[:,None]
		
		rms_u = np.zeros(10)
		rms_l = np.zeros(10)
		
		for i in range(gamel):
			a = diff[i]
			rms_u[i] = np.sqrt(np.average( a[np.where(a>0)[0]]**2   ))
			rms_l[i] = np.sqrt(np.average( a[np.where(a<0)[0]]**2   ))
				
		return  rms_u, rms_l
		
	elif (x.shape[0]!=4):
		return mean_contri(x.T)
		
	else:	
		print ("This is not what the function does")
		return None

'''
x = np.arange (4,1000)

y = (x-3.)*(x-4.)*(x-5.)
z = (x-1.)*(x-2.)*x
		
		
plt.plot (x,y/z)
ng = 40
plt.plot (x,(y/z)**(ng*(ng-1)/2.))
ng = 30
plt.plot (x,(y/z)**(ng*(ng-1)/2.))
ng = 20
plt.plot (x,(y/z)**(ng*(ng-1)/2.))
ng = 10
plt.plot (x,(y/z)**(ng*(ng-1)/2.))
plt.show()
		


from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt



sim = 'random'  #Other option is 1

M_real = np.load('Testing_fits/random_m_' +sim +'_g_mvals.npy')
G_real = np.load('Testing_fits/random_m_' +sim +'_g_gvals.npy')

Ng = len(G_real)

datadir = 'fitdata_final/Results/'

params = np.load(datadir+sim+'params2k_m_gi_fit.npy')
M_fit = np.zeros(M_real.shape)
G_fit = np.zeros(M_real.shape)

for i in range(Ng):
    G_fit[4*i:4*i+4] = params[i][4]*np.ones (4)
    M_fit [4*i:4*i+4]  = params[i][0:4]

 
 
# Creating dataset
z = np.random.randint(100, size =(50))
x = np.random.randint(80, size =(50))
y = np.random.randint(60, size =(50))
 
# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(M_fit, M_real, G_fit, color = "green")
plt.title("simple 3D scatter plot")
ax.set_xlabel('Fitted m', fontweight ='bold')
ax.set_ylabel('Actual m', fontweight ='bold')
ax.set_zlabel('$\gamma$', fontweight ='bold')
 
# show plot
plt.show()
'''
'''

f = np.arange(0,21)

def discretegaussian(mean, variance):
    dist = np.exp(-(np.arange(21) - mean) ** 2 / (2 * (variance) ** 2))
    dist /= dist.sum()
    return dist  
    
A = discretegaussian(5, 5)
B = np.zeros(21)
B[5] = 1

print (entropy(B,A))


mb = [0,20]
Kb = [1.8,10]
	
	
bounds = np.array([Kb,mb,Kb,mb,Kb,mb,Kb,mb])
print (bounds)
'''
