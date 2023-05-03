# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 21:58:16 2016
@author: jamezcua
"""
import numpy as np
from tools_old.l96 import l96num
from scipy.linalg import circulant


def natrun(Nx,tmax):
 xmin = 0; xmax = Nx-1; dx = 1
 x = np.arange(xmin,xmax+1,dx)
 forc = 8.0;    dt = 0.025
 ttra = np.arange(0,1+dt,dt)
 u0tra = forc*np.ones((Nx))
 u0tra[0] = u0tra[0]+5
 utra = l96num(x,ttra,u0tra)  #spinup of 2 timesteps from [13,8,...8]?
 t = np.arange(0,tmax+dt,dt)
 u0t = utra[20,:]
 ut = l96num(x,t,u0t)
 ug0 = (utra[30,:] + ut[40,:])/2.0 #whats this?
 return x,t,ut,ug0


def getBc(Nx):
 Bc_row = np.zeros((Nx))
 Bc_row[0] = 3.9338
 Bc_row[1] = 1.3789; Bc_row[-1] = 1.3789
 Bc_row[2] = 0.4646; Bc_row[-2] = 0.4646
 Bc = circulant(Bc_row)
 Gamma_Bc,v_Bc = np.linalg.eig(Bc)
 Gamma_Bc_sq = np.diag(np.real(np.sqrt(Gamma_Bc)))
 Bc_sq = np.real(np.dot(v_Bc,Gamma_Bc_sq))
 return Bc, Bc_sq


def evolcov(Bc,tmat,Nx,lags):
 Bt = np.empty((Nx,Nx,lags))
 B0t = np.empty((Nx,Nx,lags))
 for j in range(lags):
  aux = np.dot(Bc,tmat[:,:,j].T) # B0t= B*M(time j)
  B0t[:,:,j] = aux    
  aux = np.dot(tmat[:,:,j],aux)
  Bt[:,:,j] = aux # Bt= M(time j)*B*M(time j)
 del j
 return Bt, B0t


def covfamrun(uini,Nx,lags,Bc_sq,M):
 dt = 0.025
 t = dt*np.arange(0,lags,1)  
 Nsteps = np.size(t)
 xmin = 0; xmax = Nx-1; dx = 1
 x = np.arange(xmin,xmax+1,dx)
 uref = l96num(x, t, uini)
 U = np.empty((Nsteps,Nx,M))
 for m in range(M):
  csi_gen = np.random.RandomState(m)
  innov = np.dot(Bc_sq,csi_gen.normal(0,1,Nx))
  U[:,:,m] = l96num(x, t, uini + innov)
 del m
 Pbt = np.empty((Nx,Nx,Nsteps))
 Pb0t = np.empty((Nx,Nx,Nsteps))
 # initial time
 Xpert_0 = pert_fref(np.squeeze(U[0,:,:]),uref[0,:],Nx,M)
 
 for j in range(Nsteps):
  Xpert_t = pert_fref(np.squeeze(U[j,:,:]),uref[j,:],Nx,M)
  Pbt[:,:,j] = np.dot(Xpert_t,Xpert_t.T) # cov of t with t
  Pb0t[:,:,j] = np.dot(Xpert_0,Xpert_t.T) # cov of 0 with t
 del j 
 
 return U,Pbt,Pb0t


def pert_fref(U,uref,Nx,M):
 Xpert = np.empty((Nx,M))
 for m in range(M): 
  Xpert[:,m] = 1/np.sqrt(M-1) * (U[:,m] - uref)
 del m
 return Xpert





