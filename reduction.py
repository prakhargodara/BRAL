import random
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt


@njit
def discretegaussian(tokens, mean, variance):
    dist = np.exp(-(np.arange(tokens + 1) - mean) ** 2 / (2 * (variance) ** 2))
    dist /= dist.sum()
    return dist


@njit()
def statetransitions_paper(tokens, nrofplayers, ximinus, xiplus, variance):
    transitions = np.zeros(
        (tokens + 1, tokens * (nrofplayers - 1) + 1, tokens * (nrofplayers - 1) + 1))
    for action in range(tokens + 1):
        for k in range(tokens * (nrofplayers - 1) + 1):
            distmean = updatemean(action, nrofplayers, k, ximinus, xiplus, tokens)
            transitions[action, k, :] = discretegaussian(
                tokens * (nrofplayers - 1), distmean, variance)
    return transitions


@njit
def updatemean(action, nrofplayers, muold, ximinus, xiplus,tokens):
    distance = action * (nrofplayers - 1) - muold
    if distance < 0:
        return min(max(muold + ximinus * distance, 0), (nrofplayers-1)*tokens)
    else:
        return min(max(muold + xiplus * distance, 0), (nrofplayers-1)*tokens)
        
        
#print (updatemean(15,4,45,0.5,1.1,20)/3.)


@njit()
def bratpolicy_new(prior, transition, nrofsteps=9, alpha=0.4, tokens=20, nrofplayers=4, gamma=0.9, compcost=1):
    policy = np.zeros((nrofsteps, tokens + 1, (nrofplayers - 1) * tokens + 1, tokens + 1))
    values = np.zeros((nrofsteps + 1, tokens + 1,(nrofplayers - 1) * tokens + 1))
    tokenrange = np.arange(0, tokens * (nrofplayers - 1) + 1)

    for step in np.flip(np.arange(nrofsteps)):
        for prevact in range(tokens + 1):
            for k in range(tokens * (nrofplayers - 1) + 1):
            
                policy[step, prevact, k, :], values[step, prevact, k] = computeaction(transition[prevact, k, :],values[step + 1,:, :], prior,alpha, gamma, compcost,tokenrange, step, nrofsteps, tokens=20)
                                                                                      
    return policy


@njit()
def computeaction(transitionvec, values, prior, alpha, gamma, compcost, tokenrange, step, nrofsteps=9, tokens=20):
    expvals = np.zeros(tokens + 1)
    policy = np.zeros(tokens + 1)
    for act in range(tokens + 1):
        expvals[act] = transitionvec @ (gamma * values[act, :] +
                                        alpha * (tokenrange + act) + tokens - act)
    
    bestact = np.argmax(expvals)
    optpol = np.zeros(tokens + 1)
    optpol[bestact] = 1
    a = kldiv(optpol, prior)
    
    if compcost > 0:
        if (a < compcost):
            policy = optpol

        else:
            ub = 500.0 / (tokens * 1.6 * (1 / (1 - gamma)))
            beta = bisection(computationalcost, 0, ub,expvals[:], compcost, tokens, prior)
            policy = np.exp(beta * expvals + np.log(prior) -beta*expvals.max())

            policy /= np.sum(policy)
    else:

        policy = prior
    valuessss =     policy @ expvals
            

    return policy, valuessss


@njit
def cdf(dist):
    return dist.cumsum()


@njit
def sample(cdf):
    x = random.random()
    return np.argmax(cdf > x)



@njit
def computationalcost(b, expvals, rationality, tokens, prior):
    retval = 0

    polarr = np.exp(b * expvals + np.log(prior) - b*expvals.max())
    polarr /= np.sum(polarr)

    
    ##Zero corrections
    AAA = np.log (polarr/prior)
    AAA[AAA == -np.inf] = -10**10
    retval = np.sum(polarr * AAA)
        

    return retval - rationality

@njit
def kldiv(p, q):
    AAA = np.log(p/q)
    AAA [AAA == -np.inf] = -10**10
    result = np.sum (p * AAA)
    return result


@njit
def bisection(f, lb, ub, expvals, rationality, tokens, prior):
    tol = 0.01
    err = 1
    niter = 0
    result = 0

    f2 = f(ub, expvals, rationality, tokens, prior)

    
    if(f2 < 0 and np.abs(f2)<tol):
        return ub
    elif (f2 < 0):
        return bisection(f, ub, 2*ub, expvals, rationality, tokens, prior)
        
    #if (f2<0):
        #return ub

    while (err > tol and niter < 100):
        midpoint = (lb + ub) / 2
        f1 = f(lb, expvals, rationality, tokens, prior)
        #print (f1)
        f2 = f(ub, expvals, rationality, tokens, prior)



        fmid = f(midpoint, expvals, rationality, tokens, prior)
        lb += (np.sign(f1) * np.sign(fmid) + 1) / 2.0 * (midpoint - lb)
        ub += (np.sign(fmid) * np.sign(f2) + 1) / 2.0 * (midpoint - ub)
        err = np.abs(fmid)
        result = midpoint
        niter += 1
    return result
    
