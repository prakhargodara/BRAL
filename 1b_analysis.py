'''
Generate random trajectories of a fixed length with varying variances.
See average contri in the next round as a func of variance.
'''
import numpy as np
from reduction import *
import matplotlib.pyplot as plt
import time
import sys
import time
from numba import float64, int64
from numba import types
from multiprocessing import Pool
from mean_rms import*
from opt_xipm_gam_past import *
import numpy as np
import scipy.optimize as opt
from scipy.optimize import fsolve
from multiprocessing import Pool
##Generate trajectories

mean = 10
length = 20
players = 4


bets = np.zeros ((length+1,players),dtype = int)

K = 3.*np.ones (players)    
xip = 0.1*np.ones (players)
xim = 0.5*np.ones (players)
gamma_p = 0.9*np.ones (players)
priors = np.zeros ((players,21))
for i in range (players):
    priors[i] = discretegaussian(20, 10, 5)
##Find optimal xipm


policies = []
for i in range (players):
        #print (i)
        transitions = statetransitions_paper(params.tokens, players, xim[i], xip[i], 3)
        policies.append (bratpolicy_new (priors[i], transitions, nrofsteps = 2,nrofplayers=players ))


        
VAR = np.linspace (0.5,3,15)
         


#for blab in range (len(VAR)):
def para(blab):
    print (blab)
    iterations = 100
    av_bets = np.ones ((iterations,players))
    XIP = np.ones ((iterations,players))
    XIM = np.ones ((iterations,players))
    var = VAR[blab]
    p1 = discretegaussian(20, mean, var)
     
    for mmm in range (iterations):        
        for kst in range (length):
            bets[kst] = np.random.choice (np.arange(21),players,p=p1 )
        bets2 = bets.T
        learning_period = bets2
        for j in range (players):

            xip[j],xim[j] = optimal_xipm_solo (learning_period,xip[j],xim[j],gamma_p[j],j)
            transitions = statetransitions_paper(20, players, xim[j], xip[j], 3.)
            policies[j] = (bratpolicy_new (priors[j], transitions, nrofsteps = 50,nrofplayers=players,compcost = K[j] ))
            period = 0
            i = length
            others = np.delete (np.arange (0,players), j)
            others_bets = np.sum (bets2 [others], axis=0)[i-1]
            prevact = bets2[j,i-1]
    
        #np.random.seed (int(time.time()+i*j))
            probabilities = policies[j][period,prevact,others_bets]
            av_bets [mmm,j] = np.sum (probabilities*np.arange(21))
            XIP[mmm,j] = xip[j]
            XIM[mmm,j] = xim[j]

    return av_bets,XIP,XIM

ttt = time.time()

iterations = 208
VAR = np.linspace (0.1,3,iterations)
#p=Pool(16)

#AA =p.map(para, range(iterations))

#print (AA[1][3],AA[0][3])
#print ("Simulation has ended in ",time.time()-ttt,"secs.")
xxip = []
xxim = []
for k in range(len(VAR)):
	#AA = np.load('Testy/rand_4k_9_incm_{}.npy'.format(k))
	AA = np.load('Testy/rand_4k_5_multi_{}.npy'.format(k))
	avbets = AA[0]
	zxip = np.array(AA[1])
	zxim = np.array(AA[2])
	
	xxim.append(AA[2])
	xxip.append(AA[1])
	
	
	
	#x = np.linspace (-1.5,1.5,100)
	#y = np.zeros (len(x))
	#plt.hist2d (zxip.flatten(),zxim.flatten(),bins=40,range = [[x.min(), x.max()], [x.min(), x.max()]])
	#plt.xlim(-1.6,1.6)
	#plt.ylim(-1.6,1.6)
	#plt.plot (x,y)
	#plt.plot (y,x)
	#plt.show()
	#plt.plot (VAR[k], np.average (avbets), 'ko')
	plt.plot (VAR[k], np.average (zxip**2 + zxim**2),'ko')
	#np.average (np.arctan(zxip,zxim))
	#np.save(datadir+'/position{0}.npy'.format(k+sn*iterations),result_pos)
	#np.save(datadir+'/velocity{0}.npy'.format(k+sn*iterations),result_vel)
	#np.save(datadir+'/partnum{0}.npy'.format(k),num)
	
plt.show()


xxim = np.array (xxim)#.flatten()
xxip = np.array (xxip)#.flatten()

