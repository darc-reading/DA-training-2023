import numpy as np
from scipy.optimize import fsolve
from tools.hybrid.l96 import l96num
from tools.hybrid.etkf16 import etkf_l96

def envar(Xt,t,x,R,invR,H,y,period_obs,obsperwin,gridobs,nx_obs,Bsq,\
    lam,Lxx,Lxy,locenvar,M,invBc=None,loc_obs=None,invQ=None,Qsq=None,\
    noiseswitch=0,scwc='sc',maxmodes_in=None):                
                
    compute_qt = 0
    
    dt = t[1]-t[0]
    winlen = obsperwin*period_obs
    taux = dt*np.arange(0.0,winlen+1,1)    
    nsteps = np.size(t)
    nx = np.size(x)
    
    winnum = int((nsteps-1)/winlen)
    
    # Precreate the arrays for background and analysis
    np.random.seed(0)

    FreeEns = np.zeros((nsteps,nx,M))
    
    VarEns = np.zeros((nsteps,nx,M))
    VarEns_ = np.zeros((nsteps,nx))

    x_b = np.empty((nsteps,nx,1));  x_b.fill(np.nan)
    x_a = np.empty((nsteps,nx,1));  x_a.fill(np.nan)

    # Initial Condition for First Guess of First Window

    xold = Xt + np.dot(Bsq,(np.random.randn(nx)).T)
    
    xold = np.reshape(xold,(nx,1))
    
    ensrun = np.empty((winlen+1,nx,M)); ensrun.fill(np.nan)
    for m in range(M):
        ensrun[0,:,m] = Xt + np.dot(Bsq,(np.random.randn(nx)).T)
    xoldens = ensrun[0,:,:]
    
    if scwc=='wc' and compute_qt==1:
        ensrun_i = np.empty((winlen+1,nx,M)); ensrun_i.fill(np.nan)
        ensrun_i[0,:,:] = ensrun[0,:,:] 
        xoldens_i = ensrun_i[0,:,:]
    
    if locenvar==0:
        maxmodes = None
        Lxsq = None;   Lysq = None;
    
    if locenvar==1:
        if scwc=='sc' or scwc=='wc':
            if maxmodes_in == None:
                maxmodes = 9 # (1+2n harmonics)
            if maxmodes_in != None:
                maxmodes = maxmodes_in # (1+2n harmonics)
    
        Gamma,C = np.linalg.eig(Lxx)        
        
        ind_Gamma = np.flipud(np.argsort(Gamma))
        ind_Gamma = ind_Gamma[0:maxmodes]        
        Gamma = Gamma[ind_Gamma]
        C = C[:,ind_Gamma] 
        
        Gamma_sq = np.sqrt(Gamma)        
        Lxsq = np.dot(C,np.diag(Gamma_sq))        
        Lysq = np.dot(H,Lxsq)
                                    
    for i in range(winnum):
        # Get the observations for this window
        # print('winnum =',i)
        yaux = y[obsperwin*i:obsperwin*(i+1),:]
        
        ## First Guess for 4DEnVar
        if scwc=='sc':
            noisesw_4denv = noiseswitch
        if scwc=='wc':
            #noisesw_4denv = noiseswitch
            noisesw_4denv = 0
            
        xb_new = l96num(x,taux,xold)
        x_b[winlen*i:winlen*(i+1)+1,:,0] = xb_new

        # Free ensemble
        for m in range(M):
            xnew_aux = l96num(x,taux,xoldens[:,m])
            ensrun[:,:,m] = xnew_aux
            del xnew_aux
        FreeEns[winlen*i:winlen*(i+1)+1,:,:] = ensrun
        Xfree = ensrun
        xfree_bar = np.mean(ensrun,2)
        Xfree_pert = np.empty((winlen+1,nx,M))
        for m in range(M):
            Xfree_pert[:,:,m] = 1/(M-1)**0.5 * (Xfree[:,:,m] - xfree_bar)
        Xfreei_pert = None
        
        # LETKF
        opcini = 2; 
        Xb_e,xb_e,Xa_e,xa_e = etkf_l96(xoldens,taux,x,M,nx_obs,H,R,yaux,\
                              period_obs,lam,Lxy,opcini)
        del Xb_e, xb_e
        VarEns[winlen*i:winlen*(i+1)+1,:,:] = Xa_e
        VarEns_[winlen*i:winlen*(i+1)+1,:] = xa_e
        
        seed_b = None
        # 4DENVAR
        xa0 = one4denvar(taux,x,period_obs,obsperwin,yaux,loc_obs,H,invR,M,nx,nx_obs,\
              noiseswitch,invQ,Qsq,xb_new,seed_b,Xfree_pert,Xfreei_pert,locenvar,Lxsq,Lysq,\
              Lxx,scwc,maxmodes,compute_qt)
        if scwc=='sc':
            xa0 = np.reshape(xa0,(nx,1))
            xa_new = l96num(x,taux,xa0,noiseswitch,Qsq,seed_b)
        if scwc=='wc':
            xa_new = xa0
        x_a[winlen*i:winlen*(i+1)+1,:,0] = xa_new
        
        # create new initial conditions for 4denvar
        xold = np.reshape(xa_new[-1,:],(nx,1))
        # create new initial conditions for ETKF and free run
        for m in range(M):
            aux = Xa_e[-1,:,m] - xa_e[-1,:] + xa_new[-1,:]
            xoldens[:,m] = aux
            if scwc=='wc' and compute_qt==1:
                xoldens_i[:,m] = xoldens[:,m]
                
    return x_a, x_b, VarEns, VarEns_, FreeEns


### ----------------------------------------------------------------------------------------
def one4denvar(taux,x,period_obs,obsperwin,yaux,loc_obs,H,invR,M,nx,nx_obs,\
    noiseswitch,invQ,Qsq,xb,seed_b,Xfree_pert,Xfreei_pert,locenvar,Lxsq,Lysq,\
    Lxx,scwc,maxmodes,compute_qt): # Need to add outer vars if needed
    #"The 4DVar algorithm for one assimilation window."
    #nsteps = np.size(taux)
    outerloops = 1
    
    if outerloops>1:    
        if locenvar==0:
            Ux,sx,VTx = np.linalg.svd(np.squeeze(Xfree_pert[0,:,:]),full_matrices=False)
            
        if locenvar==1:
            Bsam = np.dot(Xfree_pert[0,:,:],Xfree_pert[0,:,:].T)
            Bsam = Bsam * Lxx
            Gamma, U = np.linalg.eig(Bsam)
            Gamma = np.real(Gamma)
            Gamma = Gamma.clip(min=0);  ind = Gamma>0; 
            Gamma[ind] = Gamma[ind]**(-1);  Gamma = np.diag(Gamma)
            invBsam = np.dot(U,np.dot(Gamma,U.T))

    
    for jotl in range(outerloops):
        xg0 = np.squeeze(xb[0,:])
        d0 = np.empty((obsperwin,nx_obs));  d0.fill(np.nan)
        Ytt = np.empty(shape=(obsperwin,nx_obs,M))
        
        for j in range(obsperwin):
            jobs = period_obs*(j+1)
            d0[j,:] = yaux[j,:] - np.dot(H,xb[jobs,:])
            # just because H is linear we can do the following    
            Ytt[j,:,:] = np.dot(H,Xfree_pert[jobs,:,:])
            del jobs
        del j
            
        # The gradient
        def gradJ(deltav): 
            # No localisation
            if scwc=='sc':
                if  locenvar==0:
                    deltav = np.reshape(deltav,(M,)) #
                    # The background term
                    if jotl == 0:
                        gJ = deltav
                    if jotl >= 1:
                        aux = np.dot(Ux.T,(xg0-xb[-1,:]))
                        aux = np.dot(np.diag(sx**(-1)), aux)
                        aux = np.dot(VTx.T,aux)
                        gJ = deltav + aux
                        del aux
                    # The observation error term, evaluated at different times
                    for i in range(obsperwin):
                        aux = np.dot(np.squeeze(Ytt[i,:,:]),deltav)
                        aux = d0[i,:] - aux
                        aux = np.dot(invR,aux)
                        aux = -np.dot(np.squeeze(Ytt[i,:,:]).T,aux)
                        gJ = gJ + aux        
                        del aux
                    del i
                    
                # With localisation
                if  locenvar==1: 
                    deltav = np.reshape(deltav,(M*maxmodes,)) # Fixing Annoyances aka fsolve
                    # The background term
                    if jotl==0:
                        gJ = deltav
                        
                    if jotl>=1:
                        gJ = deltav                     
                        z_aux = np.dot(invBsam,(xg0-xb[-1,:]))
                        for m in range(M):
                            aux = np.squeeze(Xfree_pert[0,:,m],z_aux)                            
                            gJ[m*nx_obs:(m+1)*nx_obs] = gJ[m*nx_obs:(m+1)*nx_obs] \
                                                        + np.dot(Lxsq.T,aux)
                            del aux
                        del m, z_aux
                        
                    # The observation error term, evaluated at different times
                    for i in range(obsperwin):
                        z = np.zeros((nx_obs,))
                        for m in range(M):
                            aux = np.dot(Lysq,deltav[m*maxmodes:(m+1)*maxmodes])
                            aux = Ytt[i,:,m] * aux
                            z = z + aux
                            del aux
                        z = np.dot(invR, d0[i,:]-z)
                        gJo_i = np.zeros((M*maxmodes,))
                        for m in range(M):
                            aux = Ytt[i,:,m] * z
                            aux = np.dot(Lysq.T,aux)
                            gJo_i[m*maxmodes:(m+1)*maxmodes] = -aux
                            del aux
                            
                        gJ = gJ + gJo_i
                        del gJo_i
                    del i
                    
# ------------------------------
            if scwc=='wc':
                if  locenvar==0:
                    v = np.reshape(deltav,(M*(1+obsperwin),)) # Fixing Annoyances aka fsolve
                    gJ = np.empty((M*(1+obsperwin),))
                    # the background and model error
                    dyt = np.empty((obsperwin,M))
                    for j in range(obsperwin):
                        aux = v[0*M:1*M] + v[(j+1)*M:(j+2)*M]
                        aux = d0[j,:] - np.dot(Ytt[j,:,:],aux)
                        aux = np.dot(invR,aux)
                        dyt[j,:] = np.dot(Ytt[j,:,:].T, aux)
                        del aux
                    del j

                    gJ[0*M:1*M] = v[0*M:1*M] - np.sum(dyt,0)
                    
                    for j in range(obsperwin):
                        jobs = period_obs*(j+1)
                        
                        if compute_qt==0:                
                            invQt = invQ#/jobs
                            aux = np.dot(Xfree_pert[jobs,:,:], v[(j+1)*M:(j+2)*M])
                            aux = np.dot(Xfree_pert[jobs,:,:].T, np.dot(invQt,aux))
                        
                        if compute_qt==1:
                            Deltaip = Xfreei_pert[jobs,:,:] - Xfree_pert[jobs,:,:]
                            auxinv = np.linalg.pinv(np.dot(Deltaip,Deltaip.T),rcond=1e-8)
                            aux = np.dot(Xfree_pert[jobs,:,:] , v[(j+1)*M:(j+2)*M])
                            aux = np.dot(auxinv,aux)
                            aux = np.dot(Xfree_pert[jobs,:,:].T,aux)
                             
                        gJ[(j+1)*M:(j+2)*M] = aux - dyt[j,:]                        
                        del aux
                    del j

# ------------------------                    
                # With localisation
                if  locenvar==1: 
                    v = np.reshape(deltav,(maxmodes*M*(1+obsperwin),)) # Fixing Annoyances aka fsolve
                    gJ = np.empty((maxmodes*M*(1+obsperwin),))
                    
                    # the background
                    gJ[0*M*maxmodes:1*M*maxmodes] = v[0*maxmodes*M:1*maxmodes*M]
                    
                    # model error
                    for j in range(obsperwin):
                        jobs = period_obs*(j+1)
                        mod_error = np.zeros((maxmodes*M))
                        z = np.zeros((nx))
                        for m in range(M):
                            aux = v[(j+1)*maxmodes*M:(j+2)*maxmodes*M]                            
                            aux = np.dot(Lxsq,aux[m*maxmodes:(m+1)*maxmodes])
                            aux = Xfree_pert[jobs,:,m] * aux
                            z = z + aux
                            
                            del aux
                        del m
                        z = np.dot(invQ,z)
                        
                        for m in range(M):
                            aux = Xfree_pert[jobs,:,m] * z
                            mod_error[maxmodes*m:maxmodes*(m+1)] = np.dot(Lxsq.T,aux)
                            del aux
                        del m
                        gJ[(j+1)*maxmodes*M:(j+2)*maxmodes*M] = mod_error
                        del mod_error
                    del j, z
                    
                    # observations
                    for j in range(obsperwin):
                        z = np.zeros((nx_obs))
                                                
                        for m in range(M):
                            aux0 = v[0*maxmodes*M:1*maxmodes*M]
                            aux0 = aux0[m*maxmodes:(m+1)*maxmodes]
                            auxt = v[(j+1)*maxmodes*M:(j+2)*maxmodes*M]
                            auxt = auxt[m*maxmodes:(m+1)*maxmodes]
                            aux = aux0+auxt
                            del aux0, auxt
                            aux = np.dot(Lysq,aux)
                            aux = Ytt[j,:,m]*aux
                            z = z + aux
                            del aux
                        del m
                                                    
                        z = np.dot(invR,d0[j,:]-z)
                        incr = np.zeros((maxmodes*M))
                        
                        for m in range(M):
                            aux = Ytt[j,:,m] * z
                            obs_error = np.dot(Lysq.T,aux)
                            incr[m*maxmodes:(m+1)*maxmodes] = -obs_error
                            del obs_error
                        del m

                        gJ[0*maxmodes*M:1*maxmodes*M] = gJ[0*maxmodes*M:1*maxmodes*M] + incr                           
                        gJ[(j+1)*maxmodes*M:(j+2)*maxmodes*M] = gJ[(j+1)*maxmodes*M:(j+2)*maxmodes*M] + incr
                               
                        del z, incr                        
                        
            return gJ.flatten()
            
# -----------------------------------            
        if scwc=='sc':    
            if  locenvar==0:
                v0 = np.zeros((M,))
                v = fsolve(gradJ,v0,xtol=1e-6,maxfev=20)        
                xa = np.squeeze(xb[0,:]) + np.dot(Xfree_pert[0,:,:],v)    
            if  locenvar==1:
                v0 = np.zeros((M*maxmodes,))
                v = fsolve(gradJ,v0,xtol=1e-6,maxfev=20)
                deltax = np.zeros((nx,))
                for m in range(M):
                    aux = np.dot(Lxsq,v[m*maxmodes:(m+1)*maxmodes])
                    aux = Xfree_pert[0,:,m] * aux
                    deltax = deltax + aux
                    del aux
                xa = xg0 + deltax        
            if jotl<outerloops-1:
                xb,seed_b = l96num(x,taux,xa,noiseswitch,Qsq,seed_b)

#-------------------------------------        
        if scwc=='wc':
            if  locenvar==0:
                v0 = np.zeros((M*(1+obsperwin),))
                v = fsolve(gradJ,v0,xtol=1e-6,maxfev=10)
                xa0 = xb[0,:] + np.dot(Xfree_pert[0,:,:],v[0*M:1*M])
                xa,seed_a = l96num(x,taux,xa0,0,Qsq)
                for jsteps in range(obsperwin):
                    jobs = period_obs * (jsteps+1)
                    xa[jobs,:] = xb[jobs,:] + np.dot(Xfree_pert[0,:,:],\
                                 v[(jsteps+1)*M:(jsteps+2)*M])
                del jsteps, jobs
                
            if  locenvar==1:
                v0 = np.zeros((maxmodes*M*(1+obsperwin),))             
                v = fsolve(gradJ,v0,xtol=1e-6,maxfev=10)
                deltax0 = np.zeros((nx,))
                auxv0 = v[0*maxmodes*M:1*maxmodes*M]
                for m in range(M):
                    aux = np.dot(Lxsq,auxv0[m*maxmodes:(m+1)*maxmodes])
                    aux = Xfree_pert[0,:,m] * aux
                    deltax0 = deltax0 + aux
                    del aux
                del m, auxv0
                xa0 = xb[0,:] + deltax0 
                xa,seed_a = l96num(x,taux,xa0,0,Qsq)
                
                for jsteps in range(obsperwin):
                    deltaxt = np.zeros((nx,))
                    auxvt = v[(jsteps+1)*maxmodes*M:(jsteps+2)*maxmodes*M]
                    jobs = (1+jsteps) * period_obs
                    
                    for m in range(M):
                        aux = np.dot(Lxsq,auxvt[m*maxmodes:(m+1)*maxmodes])
                        aux = Xfree_pert[jobs,:,m] * aux
                        deltaxt = deltaxt + aux
                        del aux
                    del m, auxvt
                    xa[jsteps,:] = xb[jsteps,:] + deltaxt
                    del jobs
                    
                del jsteps

            xb = xa
                
    return xa

#############



