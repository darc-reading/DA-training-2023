# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:39:29 2015
@author: jamezcua
"""

import numpy as np
import random

def transmat_l96(u0,t,x,noiseswitch=0,Qsq=0,ain=None):
    """The transition matrix required for the TLM and the adjoint.
    Inputs:  - u0
             - t
             - x
    Outputs: - M, the transition matrix for small perturbations from
               the initial state to the evolved state [3 x 3 x len(t)]"""
    nsteps = np.size(t)    
    nx = np.size(x)
    dt = t[1]-t[0]
    dx = x[1]-x[0]
    
    if noiseswitch==0:
        aout = ain
    if noiseswitch==1:
        if ain==None:
            aout = []
        if ain!=None:
            aout = ain

    M0 = np.eye(nx)
    u = np.empty((nsteps,nx))
    u.fill(np.nan)
    u[0,:] = u0
    
    M = np.empty((nx,nx,nsteps))
    M.fill(np.nan)
    M[:,:,0] = M0

    for i in range(nsteps-1): # for each time
        uaux,M[:,:,i+1] = integr(u[i,:],M[:,:,i],dt,dx,nx)
        if noiseswitch==0:
            innov = np.zeros((nx))            
        if noiseswitch==1:
            if ain==None:
                a = np.int(random.uniform(1,100000))
                aout.append(a)
            if ain!=None:
                a = ain[i]
            csi_gen = np.random.RandomState(i+a)
            innov = csi_gen.normal(0,1,nx)
            innov = np.dot(Qsq,np.reshape(innov,(nx,1)))
            innov = np.reshape(innov.real,(nx,)) 
        u[i+1,:] = uaux + innov
                
    return u, M, aout


def integr(xin,Min,dt,dx,nx):
    # The integration is for both the model and the TLM at the same time!
    Varsold = np.concatenate((xin,Min.flatten()))
    k1 = faux(Varsold,dx,nx)
    k2 = faux(Varsold+1/2.0*dt*k1,dx,nx)
    k3 = faux(Varsold+1/2.0*dt*k2,dx,nx)
    k4 = faux(Varsold+dt*k3,dx,nx)
    Varsnew = Varsold + 1/6.0*dt*(k1+2*k2+2*k3+k4)
    xout = Varsnew[:nx]
    Mout = np.reshape(Varsnew[nx:],(nx,nx))
    return xout,Mout


def faux(Varsin,dx,nx):
    dudt = f(Varsin[:nx],dx,nx)
    F = tlm(Varsin[:nx],dx,nx)
    dM = np.dot(F,np.reshape(Varsin[nx:],(nx,nx)))
    dxaux = np.concatenate((dudt,dM.flatten()))
    return dxaux


def f(uin,dx,nx):
    forc = 8.0
    u = np.zeros((nx+3))    
    u[2:nx+2] = uin
    u[1] = uin[-1]    
    u[0] = uin[-2]
    u[nx+2] = uin[0]
    evalf = u[1:nx+1] * (-u[0:nx]+u[3:nx+3]) - u[2:nx+2] + forc
    return evalf
    

def tlm(u,dx,nx):
    F = np.zeros((nx,nx))
    for j in range(nx):
        F[j,(j-2)%nx] = -u[(j-1)%nx] 
        F[j,(j-1)%nx] = -u[(j-2)%nx] + u[(j+1)%nx]
        F[j,j%nx] = -1
        F[j,(j+1)%nx] = u[(j-1)%nx]
    return F
    

