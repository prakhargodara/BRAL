from sim_gam_past import *
from multiprocessing import Pool
import multiprocessing as mp



tokens=20
 
n_players = 4

alpha = 1.6/float(n_players)
print ('alpha = ',alpha)

def discretegaussian(mean, variance):
    dist = np.exp(-(np.arange(tokens + 1) - mean) ** 2 / (2 * (variance) ** 2))
    dist /= dist.sum()
    return dist    
    
    
#for j in range (4,9):
	#a = np.load("A_{}_players.npy".format(j))/90.
	#plt.plot (a)
#plt.show()
n=100
gamma = 0.9
gam_p = 0.9

mm = 10.*np.ones(n_players)
cc = 4.5*np.ones(n_players)
gg = gamma*np.ones(n_players)
ib = 10*np.ones(n_players,dtype=int)#np.random.randint (0, 20, size = (n_players))
xip = 0.1*np.ones(n_players)
xim = 0.5*np.ones(n_players)
gp = gam_p*np.ones(n_players)
priors = np.ones ((n_players,21))/21.
 
for i in range (n_players):
	priors[i] = discretegaussian (mm[i],5)


#bets= simulation_nolearn (xip,xim,cc,gg,priors,ib,n)

#plt.plot (bets.T)
#plt.savefig ("Learningtests/No_learning_params{}_bets".format(int (cc[0]*10)))
#plt.show()

BB = []
def multi (i):
	np.random.seed()
	ib = np.random.randint (0, 20, size = (4))
	#print (ib)
	AA = simulation_learn (xip,xim,cc,gg,gp,priors,ib,n)
	#Change sim_gam_past to sim2
	#AA = simulation_learn (xip,xim,cc,gg,priors,LL,ib,n)
	return AA


iterations = 45
p=Pool(15)

tt = time.time()

AA = p.map(multi, range(iterations))

#BE= np.array (BE)

print ((time.time()-tt)/60., "mins")



bets = np.zeros ((n_players,n))
xipss = np.zeros ((n_players,n))
ximss = np.zeros ((n_players,n))

for i in range (iterations):
	bets += AA[i][0]

	
	xipss += AA[i][1]
	ximss += AA[i][2]

#np.save ('A_{}_players'.format(n_players),np.average(bets,axis=0))
print (bets.mean()/iterations)
plt.plot (bets.T/iterations)
#plt.savefig ("Learningtests/Learning_params{}_bets".format(int (cc[0]*10)))
plt.show()

#plt.plot (ximss.T/iterations)
#plt.savefig ("Learningtests/Learning_params{}_xim".format(int (cc[0]*10)))
#plt.show()

#plt.plot (xipss.T/iterations)
#plt.savefig ("Learningtests/Learning_params{}_xip".format(int (cc[0]*10)))
#plt.show()

#params1 = np.array ([0.9,5,10,4.2,iterations])
#np.save ('Learningtests/params1',params1)

