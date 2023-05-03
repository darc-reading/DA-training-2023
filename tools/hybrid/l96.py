# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:14:11 2015
@author: jamezcua
"""
import numpy as np
import random

def l96num(x,t,u_ini,noiseswitch=0,Qsq=0,jump=None):
    dt = t[1]-t[0]
    dx = x[1]-x[0]
    nx = np.size(x)
    nsteps = np.size(t)
    u = np.zeros((nsteps,nx+3))    
    
    # assign initial conditions       
    u_ini = np.squeeze(u_ini)
    u[0,2:nx+2] = u_ini
    
    u[0,1] = u_ini[-1]    
    u[0,0] = u_ini[-2]
    u[0,nx+2] = u_ini[0]
        
    intmet = 1
            
    # integration 
    for it in range(nsteps-1):   
        uold = u[it,:]               
        if intmet==0: # Euler forward 
            unew = eulerf(uold,dt,dx,nx)
        if intmet==1: # RK4
            unew = rk(uold,dt,dx,nx)
            
        if noiseswitch==0:
            innov = np.zeros((nx))
        if noiseswitch==1:
            a = np.int(random.uniform(1,10000))
            csi_gen = np.random.RandomState(it+a)
            innov = csi_gen.normal(0,1,nx)
            innov = np.dot(Qsq,np.reshape(innov,(nx,1)))
            innov = np.reshape(innov.real,(nx,)) # This is the error for each variable at each grid point at each time step.
        if noiseswitch==2:
            innov = jump
        
        u_new = unew + innov
        u[it+1,2:nx+2] = u_new
        u[it+1,1] = u_new[-1]
        u[it+1,0] = u_new[-2]
        u[it+1,nx+2] = u_new[0]
        #print("it",it)
    
    u = u[:,2:nx+2]
    return u


#########
def eulerf(uold,dt,dx,nx,k):
    unew = uold[2:nx+2] + dt * f(uold,dx,nx)
    return unew


##########
def rk(uold,dt,dx,nx):
    k1 = np.zeros((nx+3))
    k1_aux = f(uold,dx,nx)
    k1[2:nx+2] = k1_aux
    k1[1] = k1_aux[-1]
    k1[0] = k1_aux[-2]
    k1[nx+2] = k1_aux[0]
    uold1 = uold + dt/2*k1    

    k2 = np.zeros((nx+3))
    k2_aux = f(uold1,dx,nx)
    k2[2:nx+2] = k2_aux
    k2[1] = k2_aux[-1]
    k2[0] = k2_aux[-2]
    k2[nx+2] = k2_aux[0]
    uold2 = uold + dt/2*k2    
    
    k3 = np.zeros((nx+3))
    k3_aux = f(uold2,dx,nx)
    k3[2:nx+2] = k3_aux
    k3[1] = k3_aux[-1]
    k3[0] = k3_aux[-2]
    k3[nx+2] = k3_aux[0]
    uold3 = uold + dt*k3    
    
    k4 = np.zeros((nx+3))
    k4_aux = f(uold3,dx,nx)
    k4[2:nx+2] = k4_aux
    k4[1] = k4_aux[-1]
    k4[0] = k4_aux[-2]
    k4[nx+2] = k4_aux[0]

    unew = uold[2:nx+2] + \
           dt/6 * (k1[2:nx+2] + 2*k2[2:nx+2] + 2*k3[2:nx+2] + k4[2:nx+2])
     
    return unew
     
#########     
def f(uold,dx,nx):
    forc = 8.0;
    evalf = uold[1:nx+1] * (-uold[0:nx]+uold[3:nx+3]) - uold[2:nx+2] + forc
    
    return evalf