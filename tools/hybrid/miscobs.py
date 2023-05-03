# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:33:05 2015

@author: jamezcua
"""
import numpy as np
import scipy as sp

def getHR(gridobs,nx,stdobs):
 if gridobs =='all':
  nx_obs = nx
 if gridobs =='1010' or gridobs=='landsea':
  nx_obs = int(nx/2)
 # Create H
 H = np.eye(nx,nx)
 # get rid of unused parts of H 
 if gridobs=='1010':
  for i in range(nx):
   if i%2==0:
    H[i,i]=0
  del i  
 if gridobs=='landsea':
  for i in range(nx):
   if i>=(nx/2) and i<nx:
    H[i,i]=0
  del i
 H = H[~np.all(H == 0, axis=1)]
        
 # location of obs    
 loc_obs = range(1,nx+1,1)
 loc_obs = np.dot(H,loc_obs)
 loc_obs = loc_obs[loc_obs>0]
 loc_obs = loc_obs-1
 loc_obs = loc_obs.astype(int)	

 R = np.eye(nx_obs)
 for i in range(nx_obs):
  R[i,i]= stdobs**2
 del i
          
 Rsq = sp.linalg.sqrtm(R);   
 invR = sp.linalg.pinv(R)

 return nx_obs, loc_obs, H, R, Rsq, invR     
    



def genobs(dt,ut,Nsteps,Nx_obs,H,period_obs,Rsq,myseed=None):
 obsnum = (Nsteps-1)/period_obs
 y = np.empty((Nsteps,Nx_obs))
 y.fill(np.nan)
 np.random.seed(myseed)   
 for i in range(int(obsnum)):
  y[(i+1)*period_obs,:]= np.dot(H,ut[(i+1)*period_obs,:]) \
                       + np.dot(Rsq,(np.random.randn(Nx_obs)).T)
 del i              
 y2 = y[np.arange(period_obs,period_obs*(int(obsnum)+1),period_obs),:]
 tobs = np.arange(period_obs,Nsteps+1,period_obs) *dt
 return tobs, y2






