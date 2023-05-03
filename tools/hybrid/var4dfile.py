import numpy as np
from tools.hybrid.l96 import l96num
from scipy.optimize import fsolve
from tools.hybrid.transmat import transmat_l96

def var4d(Xt,t,x,H,y,period_obs,obsperwin,gridobs,Bsq,invR,\
    invB=None,invQ=None,Qsq=None,noiseswitch=0,precond=1,scwc='sc'):
        
    nsteps = np.size(t)
    nx = np.size(x)
    dt = t[1]-t[0]    
    
    winnum = int((nsteps-1)/(period_obs*obsperwin))
    taux = dt*np.arange(0.0,period_obs*obsperwin+1,1)    
    
    nouterloops = 1
        
    # Precreate the arrays for background and analysis
    np.random.seed(0)
    x_b = np.empty((nsteps,nx))
    x_b.fill(np.nan)
    x_a = np.empty((nsteps,nx))
    x_a.fill(np.nan)
    
    HTinvR = np.dot(H.T,invR)
    
    # Initial Condition for First Guess of First Window
    xold = Xt
    x_b[0,:] = xold
    x_a[0,:] = xold
    
    if scwc=='sc':
        BsqTinvQBsq = None    
    if scwc=='wc':
        BsqTinvQBsq = None#np.dot(Bsq.T,np.dot(invQ,Bsq))        
    
    for j in range(winnum):
        # Get the observations for this window
        # print('winnum =',j)
        yaux = y[obsperwin*j:obsperwin*(j+1),:] # [anawin,L]
        ## First Guess
        xb0 = x_a[j*period_obs*obsperwin,:]         
        xbaux,xaaux = one4dvarPC(xb0,taux,x,yaux,H,Bsq,invR,period_obs,\
                      noiseswitch,invQ,Qsq,scwc,BsqTinvQBsq,HTinvR,nouterloops)
        x_b[j*period_obs*obsperwin+1:(j+1)*period_obs*obsperwin+1,:] = xbaux[1:,:]
        x_a[j*period_obs*obsperwin:(j+1)*period_obs*obsperwin+1,:] = xaaux

    return x_b, x_a


### ----------------------------------------------------------------------------------------
def one4dvarPC(xb0,taux,x,yaux,H,Bsq,invR,period_obs,noiseswitch,invQ,Qsq,\
               scwc,BsqTinvQBsq,HTinvR,nouterloops):
    nsteps = np.size(taux)
    nx = np.size(x)
    obsperwin,nxobs = np.shape(yaux)
    seedin = None    
    
    xg0 = xb0
    betag0 = 0
    
    for jouter in range(nouterloops):
        if scwc=='sc':
            xb,tm,seed = transmat_l96(xg0,taux,x,noiseswitch,Qsq,seedin)
            vold = np.zeros((nx,))
        
        if scwc=='wc':
            noiseswitch_b = 0
            xb,tm,seed = transmat_l96(xg0,taux,x,noiseswitch_b,Qsq,seedin)
            vold = np.zeros((nx*(1+obsperwin),))
                
        dpre = np.zeros((obsperwin,nxobs))
        
        for j in range(obsperwin):
            jobs = (j+1)*period_obs
            dpre[j,:] = yaux[j,:] - np.dot(H, xb[jobs,:])
         
        # The gradient
        def gradJ(v):
            if scwc=='sc':
                # The background term
                if jouter<1:
                    gJ = v 
                
                # The observation error term, evaluated at different times
                for j in range(obsperwin):
                    jobs = (j+1)*period_obs
                    aux = np.dot(tm[:,:,jobs] , np.dot(Bsq,v))
                    aux = dpre[j,:] - np.dot(H, aux)
                    aux = np.dot(HTinvR,aux)
                    aux = - np.dot(Bsq.T, np.dot(tm[:,:,jobs].T,aux))
                    gJ = gJ + aux
                
        
            return gJ.flatten()
            
        va = fsolve(gradJ,vold,maxfev=10)
        
        if scwc=='sc':
            xa0 = xg0 + np.dot(Bsq,va)
            xa = l96num(x,taux,xa0,noiseswitch,Qsq,seed)
            xg0 = xa0
            seedin = seed
            
    return xb,xa





