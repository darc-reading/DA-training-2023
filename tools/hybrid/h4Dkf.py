import numpy as np
from tools_old.l96 import l96num
from tools_old.etkf16 import etkf_l96
from tools_old.var4dfile import var4d


def etkf4DVar(xguess,t,x,R,invR,H,y,period_obs,obsperwin,gridobs,nx_obs,B,Bsq,\
    lam,Lxx,Lxy,locenvar,M,beta,invBc=None,loc_obs=None,invQ=None,Qsq=None,\
    noiseswitch=0,scwc='sc',maxmodes_in=None):                
                    
    dt = t[1]-t[0]
    winlen = obsperwin*period_obs
    taux = dt*np.arange(0.0,winlen+1,1)    
    nsteps = np.size(t)
    nx = np.size(x)
    
    winnum = int((nsteps-1)/winlen)
    
    # Precreate the arrays for background and analysis
    np.random.seed(0)

    Xakf = np.empty((nsteps,nx,M))
    xakf = np.empty((nsteps,nx))

    xbVar = np.empty((nsteps,nx,1));  
    xaVar = np.empty((nsteps,nx,1));  

    # Initial Condition for First Guess of First Window
    xold = xguess
    xold = np.reshape(xold,(nx,1))

    xboldens = np.empty((nx,M))
    
    for m in range(M):
        xboldens[:,m] = xguess + np.dot(Bsq,(np.random.randn(nx)).T)
    
    xaoldens = xboldens
                                        
    for i in range(winnum):
        # Get the observations for this window
        # print('winnum =',i)
        yaux = y[obsperwin*i:obsperwin*(i+1),:]
                
        # compute the Pb from the ensemble:
        Pb = computePb(xboldens,nx,M)
        
        # Compute the background for 4DVar
        xb_new = l96num(x,taux,xold,noiseswitch,Qsq)
        xbVar[winlen*i:winlen*(i+1)+1,:,0] = xb_new
        
        # LETKF
        opcini = 2; 
        Xb_e,xb_e,Xa_e,xa_e = etkf_l96(xaoldens,taux,x,M,nx_obs,H,R,yaux,\
                              period_obs,lam,Lxy,opcini)
        
        Xakf[winlen*i:winlen*(i+1)+1,:,:] = Xa_e
        xakf[winlen*i:winlen*(i+1)+1,:] = xa_e
        
        xaoldens = Xa_e[-1,:,:]
        xboldens = Xb_e[-1,:,:]
        
        del Xb_e, xb_e
        
        # Compute the hybrid matrix
        
        Ph = beta[0]*B + beta[1]*Pb*Lxx
        Phsq = msq(Ph)
    
        # 4DVAR
        xb4,xa4 = var4d(np.squeeze(xold),taux,x,H,yaux,period_obs,\
                        obsperwin,gridobs,Phsq,invR)

        xaVar[winlen*i:winlen*(i+1)+1,:,0] = xa4        
        # create new initial conditions for 4denvar
        xold = xa4[-1,:]
                
    return xbVar, xaVar, Xakf, xakf
    

def computePb(X,Nx,M):
 xmean = 1.0/M * np.dot(X,np.ones((M)))
 Xpert = np.empty((Nx,M))
 for m in range(M): 
  Xpert[:,m] = 1/np.sqrt(M) * X[:,m] - xmean
 del m
 Pb = np.dot(Xpert,Xpert.T)
 return Pb


def msq(B):
 Gamma,v = np.linalg.eig(B)
 Gamma_sq = np.diag(np.real(np.sqrt(Gamma)))
 B_sq = np.real(np.dot(v,Gamma_sq))
 return B_sq

### ----------------------------------------------------------------------------------------

#############



