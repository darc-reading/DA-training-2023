from scipy.linalg import inv, sqrtm
import numpy as np
import scipy as sp
import random
import time
#from joblib import Parallel, delayed
from tools_old.l96 import l96num

def etkf_l96(Xt,t,x,M,nx_obs,H,R,y,period_obs,lam,locmatrix,\
    opcini=1,adap=0,rho0=0.05,noiseswitch=0,Qsq=None,smooth=0):
    '''  Xt - Initial point truth
      nsteps - total time steps
       x_obs - Observations
           M - Ensemble Members
          nx - grid size
           H - observation matrix
           R - observational error covariance matrix
         loc - localisation
    11. Changed to new model
    '''
    nx = np.size(x)
    dt = t[1]-t[0]
    nsteps = np.size(t)
    obsnum = int((nsteps-1)/period_obs)    
    
    # Precreate arrays for background and analysis
    Xb = np.empty(shape=(nsteps+1,nx,M))
    Xa = np.empty(shape=(nsteps+1,nx,M))
    xold = np.empty(shape=(nx,M))
    #np.random.seed(20000)
    rhoa = np.empty(shape=(obsnum+1,nx))
    rhoa[0,:] = rho0 


    # Generate initial conditions  
    for j in range(M):
        if opcini==0:
            Xb[0,:,j] = Xt[0,:] + np.random.randn(nx).T
        if opcini==1:
            Xb[0,:,j] = Xt + np.random.randn(nx).T
        if opcini==2:
            Xb[0,:,j] = Xt[:,j]
        
    Xa[0,:,:]=Xb[0,:,:] # Background = Analysis at t=0 as no observations        
    
    # Getting the R-localization weights
    # Note that the dimensions are [nx,nx_obs], i.e. it lives in physical space
    
    ## Evolution and Assimilation cycles
    #First step is to evolve from the analysis from the prior window
    #taux = np.arange(0.0,period_obs*dt+dt,dt)
    taux = dt*np.arange(0.0,period_obs+1,1)    
 
    
    for i in range(obsnum):   
        xold = Xa[i*period_obs,:,:]        
        xnew = np.empty((np.size(taux),nx,M))
        ## First Guess
        for m in range(M):
            xnew[:,:,m] = l96num(x,taux,xold[:,m],noiseswitch,Qsq)
            Xb[period_obs*i+1:period_obs*(i+1)+1,:,m] = xnew[1:,:,m] # Xb = initial Xb (prior Xa) up until next observation
        
        # The new background
        rho_old = rhoa[i,:]
        xa_aux,rho_new = etkf(xnew,period_obs,M,R,nx,nx_obs,H,y[i,:],\
                             adap,lam,locmatrix,rho_old,smooth)
        Xa[period_obs*i:period_obs*(i+1)+1,:,:] = xa_aux
        rhoa[i+1,:] = rho_new.T
        # print (i)
        
    # The background and analysis mean
    Xa = Xa[:-1,:,:]
    Xb = Xb[:-1,:,:]        
    x_b = np.mean(Xb,axis=2) #
    x_a = np.mean(Xa,axis=2) #

    return Xb,x_b,Xa,x_a


### ----------------------------------------------------------------------------------------------------------------------------------
def etkf(Xbtraj,period_obs,M,R,nx,ny,H,y,adap,lam,locmatrix,rhob,smooth):
    # nx and ny are number of gridpoints in state and obs space respectively    
    Xb = Xbtraj[-1,:,:]    
    Yb = np.empty(shape=(ny,M))
    Xatraj = np.empty((period_obs+1,nx,M))

    for m in range(M):
        Yb[:,m]=np.dot(H,Xb[:,m])

    U = np.mat(np.ones((M,M))/M)
    I = np.mat(np.eye(M))

    # Means and perts
    xb_bar = np.mean(Xb,axis=1) 
    xb_bar = np.reshape(xb_bar,(nx,1))
    Xb_pert = np.dot(Xb,(I-U))
    yb_bar = np.mean(Yb,axis=1) 
    Yb_pert = np.dot(Yb,(I-U))

    rhoa = np.empty((nx,1))
    
    indX = np.arange(nx)
    influence_dist = np.ceil(nx/6.0) # this is a rule of thumb, could be refined
    rhob = np.reshape(rhob,(nx,1))
    
    #njo = 1
    njo = -1
    par = 0
    #par = 1
    
    if par == 1:
        aux = Parallel(n_jobs=njo)(delayed(etkfpergp) (Xbtraj[:,jgrid,:],\
                period_obs,jgrid,nx,ny,y,adap,lam,influence_dist,indX,H,R,\
                Xb[jgrid,:],xb_bar[jgrid],Xb_pert[jgrid,:],Yb,yb_bar,Yb_pert,\
                locmatrix,rhob[jgrid,0],M,smooth) for jgrid in range(nx))
        #return rhoag, Xag, Wag, wag

    for jgrid in range(nx):
        if par == 1:
            rhoa[jgrid] = aux[jgrid][0]
            Xatraj[:,jgrid,:] =  aux[jgrid][1]
            
        if par == 0:
            rhoag_aux, Xtrajag_aux = etkfpergp(Xbtraj[:,jgrid,:],\
                period_obs,jgrid,nx,ny,y,adap,lam,influence_dist,indX,H,R,\
                Xb[jgrid,:],xb_bar[jgrid],Xb_pert[jgrid,:],Yb,yb_bar,Yb_pert,\
                locmatrix,rhob[jgrid,0],M,smooth)
            #print rhoag_aux
            rhoa[jgrid] = rhoag_aux
            Xatraj[:,jgrid,:] = np.real_if_close(Xtrajag_aux)

    return Xatraj, rhoa


def etkfpergp(Xbtraj,period_obs,jgrid,nx,ny,y,adap,lam,influence_dist,indX,H,R,\
              Xb,xb_bar,Xb_pert,Yb,yb_bar,Yb_pert,locmatrix,rhob,M,smooth):
  
    # select the obs  
    if lam == []:
        useobsX = indX        
  
    if lam >= influence_dist: 
        useobsX = indX

    if lam < influence_dist:
        lim1 = mod2(jgrid-3.0*np.ceil(lam),nx) 
        lim2 = mod2(jgrid+3.0*np.ceil(lam),nx)

        if lim1==lim2:  
            useobsX = indX

        if lim1>lim2:
            useobsX = np.append([np.arange(lim1,nx)],[np.arange(0,lim2)]).T
            useobsX.sort()
            
        if lim1<lim2:   
            useobsX = np.arange(lim1,lim2,1).T
            useobsX.sort()

    NuseobsX = len(useobsX)
               
    H_aux = np.zeros(shape=(ny,NuseobsX))
    
    for j in range(NuseobsX):
        H_aux[:,j] = H[:,int(useobsX[j])]

    indobs_pre = np.dot(H_aux,useobsX+1)
    indobs_pre = np.reshape(indobs_pre,(ny,1))
    indobs = np.where(indobs_pre!=0)
    indobs = indobs[0]
    H_aux = H_aux[indobs,:]

    NuseobsY = len(indobs)
    Xatraj = Xbtraj
    
    # if localization killed all observations, nothing can be done
    if NuseobsY==0: 
        rhoag = rhob
        
    # if there are some observations left, then we can assimilate
    if NuseobsY!=0:
        # Trim loc and R
        locmatrix_aux = np.diag(locmatrix[jgrid,indobs])
        R_aux = np.diag(R[indobs,indobs])
        loc_invR = locmatrix_aux*inv(R_aux)
        # trim Yb_pert
        Yb_pert_aux = Yb_pert[indobs,:]
        yb_bar_aux = yb_bar[indobs]
        # trim y
        y_aux = y[indobs]
        d_aux = y_aux - yb_bar_aux
           
        # adaptive inflation
        if adap == 0:  
            rhoa_aux = rhob
 
        elif adap == 1:
            loc_tr = np.trace(locmatrix_aux)          
            
            vb = 0.05**2.0 # This is something prescribed and tuned!
            den = np.trace(np.dot(Yb_pert_aux,Yb_pert_aux.T)/(M-1.0)* loc_invR)
            alphab = (1+rhob)**2
            alphao = (np.trace( np.dot(d_aux,d_aux.T)* np.mat(loc_invR)) - loc_tr)/den
            vo = 2/np.trace(locmatrix_aux)*((alphab*den + loc_tr)/den)**2
            alphaa = (alphab*vo + alphao*vb)/(vo+vb)
            rhoa_aux = np.sqrt(alphaa)-1
        
        if rhoa_aux<0.01 or rhoa_aux>0.4:
            rhoa_aux = 0.05

        rhoag = rhoa_aux
            
        # the actual ETKF for each gridpoint
        Yb_pert_aux = (1.0 + rhoa_aux) * Yb_pert_aux
        beta = np.dot(loc_invR, Yb_pert_aux)
        beta = np.dot(Yb_pert_aux.T, beta)
        beta = (beta+beta.T)/2.0
        #beta = np.real_if_close(sqrtm(np.dot(beta.T,beta)))

        beta_ind = np.zeros((M,M))
        beta_fin = np.isfinite(beta,beta_ind)
        beta_sum = np.sum(beta_fin.flatten()) - M**2
        if beta_sum == 0:
            Gamma,C = np.linalg.eig(beta/(M-1.0))
            Gamma = np.real_if_close(Gamma)
            Gamma = Gamma.clip(min=0)
            indauc = Gamma > 10.0**(-10)
            Gamma = Gamma[indauc]
            C = C[:,indauc]
        
            Wag = np.dot(np.dot(C,np.diag((1+Gamma)**(-1/2.0))),C.T)
            Wag = np.real_if_close(Wag)
            
            aux = np.dot(loc_invR,d_aux)
            aux = np.dot(Yb_pert_aux.T,aux)
            aux = np.dot(Wag.T,aux.T)        
            wag = 1.0/(M-1) * np.dot(Wag,aux)
        
            wag = np.real_if_close(wag)  
        if beta_sum != 0:
            print ('warning')
            Wag = np.eye(M)
            wag = np.zeros((M,1))
        Xa_pert_j = (1+rhoa_aux)*np.dot(Xb_pert,Wag)
        
        xa_bar_j = xb_bar + (1.0 + rhoa_aux) * np.dot(Xb_pert, wag)

        Xag = np.zeros(shape=(1,M))
        for m in range(M):
            Xag_aux = np.real_if_close(Xa_pert_j[:,m] + xa_bar_j) 
            Xag[:,m] =  np.squeeze(Xag_aux)
               
        if smooth==0:
            Xatraj[-1,:] = Xag
        
        if smooth==1:
            for j in range(period_obs):
                Xatraj[j,:] = Smoother(Xbtraj[j,:],M,Wag,wag) 
            Xatraj[-1,:] = Xag

    return rhoag, Xatraj


## -----------------------------------------------------------------------------------------------------------------------------------
## Modular function (makes computations easier)
def mod2(x,y):
    if x!=0 and x!=y:
        if y == 0:
            m = x
        if y != 0:
            m = x - np.floor(x/y)*y

    if x==0 or x==y:
        m = y;

    return m


##------------------------------------------------------------------------------------------------------------------------------------
## Localization functions
# To get the localization weights
def getlocmat(nx,ny,H,lam,loctype):
    # nx and ny are number of gridpoints in state and obs space respectively
    # then N = 3*nx and L = varnum*ny
    indx = np.arange((nx))
    Hy = H[0:ny,0:nx]
    indy = np.dot(Hy,indx)
    dist = np.empty((nx,ny))
    dist.fill(np.nan)

    indx=np.reshape(indx,(1,nx))
    indy=np.reshape(indy,(1,ny))

    # first obtain a matrix that indicates the distance (in grid points
    # between state variables and observations
    for jrow in range(nx):
        for jcol in range(ny):
            dist[jrow,jcol] = min(abs(indx[0,jrow]-indy[0,jcol]),\
            nx-abs(indx[0,jrow]-indy[0,jcol]))
    
    # Now we create the localization matrix
    # If we want a sharp cuttof
    if loctype == 0:
        locmatrix = 1*(dist<=lam)

    # If we want something smooth, we use the Gaspari-Cohn function 
    elif loctype == 1:
        locmatrix = np.empty((nx,ny))
        locmatrix.fill(np.nan)
        locmatrix=np.reshape(locmatrix,(nx,ny))

        for j in range(ny):
            for k in range(nx):            
                locmatrix_aux = gasparicohn(dist[k,j],lam)
                locmatrix[k,j] = locmatrix_aux #gasparicohn(dist[:,j],lam)
    return locmatrix



###### The Gaspari-Cohn function
def gasparicohn(z,lam):
    c = lam/np.sqrt(0.3)
    zn = abs(z)/c
    if zn<=1.0:
        C0 = -(1.0/4.0)*(zn**5.0) + (1.0/2.0)*(zn**4.0) + (5.0/8.0)*(zn**3.0) \
             - (5.0/3.0)*(zn**2.0) + 1.0

    if zn>1.0 and zn<=2.0:
        C0 = (1.0/12.0)*(zn**5.0) - (1.0/2.0)*(zn**4.0) + (5.0/8.0)*(zn**3.0) \
             + (5.0/3.0)*(zn**2.0) - 5.0*zn + 4.0 - (2.0/3.0)*(zn**(-1.0))

    if zn>2.0:
        C0 = 0.0

    return C0

def Smoother(Xbprov,M,Wa,wa):
    U = np.mat(np.ones((M,M))/M)
    I = np.mat(np.eye(M))
    xbprov_bar = np.dot(Xbprov,np.ones((M,1))/M)
    Xbprov_pert = Xbprov*(I-U)
    Xaprov_pert = Xbprov_pert*Wa
    xaprov_bar = xbprov_bar + Xbprov_pert*wa
    Xaprov = Xaprov_pert + xaprov_bar*np.ones((1,M))
    return Xaprov


