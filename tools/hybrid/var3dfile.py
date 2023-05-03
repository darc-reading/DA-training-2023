import numpy as np
from tools_old.l96 import l96num
from scipy.optimize import fsolve

def var3d(Xt,t,x,H,y,period_obs,gridobs,B0sq,invR,\
    Qsq=None,opcini=1,invB=None,noiseswitch=0,precond=1):
        
    nsteps = np.size(t)
    nx = np.size(x)
    dt = t[1]-t[0]    
    
    winnum = int((nsteps-1)/period_obs)
    
    # Precreate the arrays for background and analysis
    #np.random.seed(0)
    x_b = np.empty((nsteps,nx))
    x_b.fill(np.nan)
    x_a = np.empty((nsteps,nx))
    x_a.fill(np.nan)

    # Initial Condition for First Guess of First Window
    if opcini == 0:
        xold = Xt[0,:] + np.dot(B0sq,(np.random.randn(nx)).T)
    if opcini == 1:
        xold = Xt
        
    x_b[0,:] = xold
    x_a[0,:] = xold
    
    #taux = np.arange(0.0,period_obs*dt+dt,dt)
    taux = dt*np.arange(0.0,period_obs+1,1)        
    for i in range(winnum):
        # Get the observations for this window
        #print('winnum =',i)
        yaux = y[i,:]
        ## First Guess
        xnew = l96num(x,taux,xold,noiseswitch)
        
        x_b[period_obs*i+1:period_obs*(i+1)+1,:] = xnew[1:,:]
        x_a[period_obs*i:period_obs*(i+1)+1,:] = xnew[:,:]
        # 3DVAR
        if precond ==1:
            xa0 = one3dvarPC(xnew[-1,:],yaux,H,B0sq,invR,nx)  
        x_a[period_obs*(i+1),:] = xa0
        # Reshaping
        xold = x_a[period_obs*(i+1),:]
        # print (i)
    return x_b, x_a



### ----------------------------------------------------------------------------------------
def one3dvarPC(xold,yaux,H,sqrtB,invR,nx):
    x0bvec = xold
    d = yaux - np.dot(H,x0bvec)    
    
    # The Cost function
    def CostFun(v):
        v = np.reshape(v,(nx,1)) 
        # The background term
        Jb = 0.5*np.dot(v.T,v)    
        # The observation error term
        aux = np.dot(sqrtB,v)
        aux = np.dot(H,aux)
        aux = d - aux
        aux2 = np.dot(invR,aux)
        Jok = 0.5*np.dot(aux.T,aux2)
        # Adding the two
        Jout = Jb + Jok     

        return Jout


    # The gradient
    def gradJ(v):
        # The background term
        gJb = v    
        # The observation error term, evaluated at different times
        gJok = np.empty((nx,))    
        aux = np.dot(sqrtB,v)        
        aux = np.dot(H,aux)
        aux = d - aux
        aux = np.dot(invR,aux)
        aux = np.dot(H.T,aux)
        aux = np.dot(sqrtB.T,aux)
        gJok = -aux
        
        # Adding the two
        gJ = gJb + gJok

        return gJ.flatten()

    #xold = np.reshape(xold,(3*nx,)) # Fixing Annoyances aka fsolve

    vold = np.zeros((nx,))
    #check = check_grad(CostFun,gradJ,xold); print('Checker =',check)

    va = fsolve(gradJ,vold) 
    if np.sum(np.isfinite(va))==nx:
     xa = x0bvec + np.dot(sqrtB,va)
    else:
     xa = x0bvec
    
    return xa


