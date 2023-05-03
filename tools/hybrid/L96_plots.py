# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:10:17 2017
@author: jamezcua
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import math
from tools_old.rmse_spread import rmse_spread 

#==========================================================
# Plot covariance 
#==========================================================

def plot_matrix(P, name='matrix'):
    fig = plt.figure()
    ax = fig.subplots(1,2)
    fig.subplots_adjust(wspace=.3)
    
    #Plot covariance
    clim = np.max(np.abs(P)) * np.array([-1,1])
    im = ax[0].pcolor(P, cmap='seismic', clim=clim)
    ax[0].invert_yaxis()
    
    #set labels
    ax[0].set_aspect('equal')
    ax[0].set_xlabel(r'$x_{i}$')
    ax[0].set_ylabel(r'$x_{j}$')
    
    #offset ticks by 0.5
    ax[0].xaxis.set_major_locator(ticker.IndexLocator(1,.5))      
    ax[0].yaxis.set_major_locator(ticker.IndexLocator(1,.5))     
    ax[0].xaxis.set_major_formatter(lambda x,pos: str(int(x)))      
    ax[0].yaxis.set_major_formatter(lambda x,pos: str(int(x)))     
    ax[0].set_title(name)
    
    #Add colorbar
    cb = plt.colorbar(im, orientation='horizontal')
    
    #Plot spectrum
    s, V = np.linalg.eig(P)
    ax[1].plot(np.arange(len(s)), np.sort(s)[::-1], 'ko')
    ax[1].set_ylim(0, 1.05*np.max(s))
    
    #Labels for spectrum
    ax[1].set_xlabel('Index')
    ax[1].set_ylabel('Eigenvalue')
    ax[1].set_title('spectrum')
    ax[1].xaxis.set_major_locator(ticker.IndexLocator(1,0))
    ax[1].grid()


# =========================================================
# Plot Lorenz 96 time sequence
# =========================================================

def plotL96 (t, xt, N, title):
  largest = max([np.abs(np.min(xt)), np.abs(np.max(xt))])
  levs = np.linspace(-largest,largest,21)
  mycmap = plt.get_cmap('BrBG',21)
  C = plt.contourf(np.arange(N), t, xt, cmap=mycmap, levels=levs)
  plt.contour(np.arange(N), t, xt, 10, colors='k', linestyles='solid')
  plt.xlabel('variable number')
  plt.ylabel('time')
  plt.title('Hovmoller diagram: ' + title)
  plt.colorbar(C,extend='both')
  plt.show()
  return
 

# =========================================================
# Plot Lorenz 96 time observations
# =========================================================

def plotL96obs(tobs, y, p, title):
  levs = np.linspace(-15,15,21)
  mycmap = plt.get_cmap('BrBG',21)
  plt.figure().suptitle(title)
  y_trans = np.transpose(y)
  C = plt.contourf(np.arange(p), tobs, y_trans, cmap=mycmap, levels=levs)
  plt.contour(np.arange(p), tobs, y_trans,  colors='k', linestyles='solid')
  plt.xlabel('ob number')
  plt.ylabel('ob time')
  plt.title(title)
  plt.colorbar(C,extend='both')
  plt.show()
  return


        
#############################################################################        
def plotL96DA_kf(t,xt,tobs,y,Nx,observed_vars,Xb,xb,Xa,xa,exp_title):
 plt.figure().suptitle('Ensemble:'+exp_title)
 for i in range(int(Nx)):
  plt.subplot(int(np.ceil(Nx/4.0)),4,i+1)
  plt.plot(t,xt[:,i],'k')
  plt.plot(t,Xb[:,i,:],'--b')
  plt.plot(t,Xa[:,i,:],'--m')
  if i in observed_vars:
   plt.autoscale(False) # prevent scatter() from rescaling axes
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
 del i    
 plt.subplots_adjust(wspace=0.7,hspace=0.3)

 plt.figure().suptitle(exp_title)
 for i in range(int(Nx)):
  plt.subplot(int(np.ceil(Nx/4.0)),4,i+1)
  plt.plot(t,xt[:,i],'k',label='truth')
  if i in observed_vars:
   plt.autoscale(False) # prevent scatter() from rescaling axes
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r',label='obs')
  plt.plot(t,xa[:,i],'m',label='analysis')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
 del i    
 plt.legend()
 plt.subplots_adjust(wspace=0.7,hspace=0.3)


# =========================================================
# Plot observations vs model observations
# =========================================================

def plotL96_obs_vs_modelobs (x_traj, y_traj, H, period_obs, exp_title):
  p        = y_traj.shape[0]
  nobtimes = y_traj.shape[1]
  
  # Set-up the arrays used to store the observations and model observations
  obs       = []
  model_obs = []

  # Loop over the times that we have observations
  for obtime in range(nobtimes):
    # What state time index does this correspond to?
    xtime = obtime * period_obs
    # Loop over the individual observations at this time
    for ob in range(p):
      # Store the observation separately
      obs.append(y_traj[ob,obtime])
      # Find the model observation
      model_obs.append(np.sum(H[ob,:] * x_traj[:,xtime]))

  # Plot the obs vs model obs
  plt.figure().suptitle(exp_title)
  plt.scatter(model_obs[:], obs[:]) 
  plt.xlabel('model observations')
  plt.ylabel('observations')
  plt.grid(True)
  plt.show()
  return




# =========================================================
# Plot histogram of values minus truth
# =========================================================
def make_histogram (nbins, xtrue_traj, x_traj, model_times, anal_times, title):
  n = xtrue_traj.shape[0]
  # Make flattened arrays of truth and analyses at analysis times
  truth = []
  x     = []
  for time in anal_times:
    tindex = find_in_array(model_times, time, 0.0000001)
    for el in range(n):
      truth.append(xtrue_traj[el,tindex])
      x.append(x_traj[el,tindex])

  Npopulation_t = len(truth)
  Npopulation_e = len(x)

  if (Npopulation_e != Npopulation_t):
    print ('Error from make_histogram: the populations of the truth and analysis are meant to be the same.')
    print (Npopulation_t, Npopulation_t)
    exit(0)

  # Convert to numpy arrays
  truth  = np.asarray(truth)
  x      = np.asarray(x)

  # Compute the error
  error  = x - truth

  # Compute the mean and standard deviation
  mean   = np.mean(error)
  std    = np.std(error)

  # Find the maximum absolute value of these differences
  maxval = max(abs(error))

  # Generate the bins
  bins   = np.linspace(-1.0*maxval, maxval, nbins+1)

  # Plot histogram
  fig, ax = plt.subplots()
  ax.hist(error, bins=bins, histtype='bar', facecolor='b')
  ax.set_title(title + '\nmean = ' + str.format('%.5f' % mean) + ', stddev = ' + str.format('%.5f' % std) + ', population = ' + str(Npopulation_t))
  ax.set_xlabel('estimate - truth')
  ax.set_ylabel('frequency')
  plt.show()
  return

# =========================================================
# Find position of an item in an array (the first occurrence)
# Failing to find the item returns zero
# =========================================================
def find_in_array (array, item, tolerance):
  pos = -1
  for p in range(len(array)):
    if abs(array[p]-item) < tolerance:
      pos = p
      break
  return pos


# =========================================================
# Plot correctness diagram (to test correctness of the TL model)
# or
# Plot gradient test
# =========================================================
def plot_log_test (x, y, xaxname, yaxname, title):
  fig, ax = plt.subplots()
  ax.plot(x, y, 'k')
  ax.set_xscale('log')
  ax.set_yscale('log')
  plt.title(title)
  plt.xlabel(xaxname)
  plt.ylabel(yaxname)
  plt.grid(True)
  plt.show()
  return



#############################################################################
def plotL96DA_var (t, xt_traj , xb_traj, xa_traj, tobs, y_traj, anal_times, H, R, exp_title):
  n = xt_traj.shape[0]
  p = y_traj.shape[0]

  for i in range(n):
    plt.figure().suptitle(exp_title + ' variable ' + str(i))
    #ax = plt.subplot(n, 1, i+1)

    # Plot vertical lines at analysis points
    for anal_time in anal_times:
      plt.axvline(x=anal_time, color='y')

    plt.plot(t, xt_traj[i,:], 'k', label='truth')
    plt.plot(t, xb_traj[i,:], 'b', label='background')
    plt.plot(t, xa_traj[i,:], 'r', label='analysis')
    # Is this state component directly observed?
    ob_component_no = -1
    for ob in range(p):
      if H[ob,i] == 1.0:
        ob_component_no = ob
    if (ob_component_no > -1):
      plt.errorbar(tobs[:], y_traj[ob_component_no,:], yerr=np.sqrt(R[ob_component_no,ob_component_no]), color='g', ls='none')

    plt.ylabel('x['+str(i)+']')
    plt.xlabel('time')
    plt.legend()
    plt.grid(True)
    plt.show()
  return
 

#####################################################
def plotL96DA_pf(t,xt,Nx,tobs,y,observed_vars,xpf,x_m):
 plt.figure().suptitle('Truth, Observations and Ensemble')
 for i in range(Nx):
  plt.subplot(np.ceil(Nx/4.0),4,i+1)
  plt.plot(t,xpf[:,i,:],'--m')
  plt.plot(t,xt[:,i],'-k',linewidth=2.0)
  plt.plot(t,x_m[:,i],'-m',linewidth=2)
  if i in observed_vars:
   plt.autoscale(False) # prevent scatter() from rescaling axes
   plt.scatter(tobs,y[:,observed_vars.index(i)],20,'r')
  plt.ylabel('x['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
 del i 
 plt.subplots_adjust(wspace=0.7,hspace=0.3)


#############################################################################
def plotpar(Nparam,tobs,paramt_time,Parama,parama):
 plt.figure().suptitle('True Parameters and Estimated Parameters')
 for i in range(Nparam):
  plt.subplot(Nparam,1,i+1)
  plt.plot(tobs,paramt_time[:,i],'k')
  plt.plot(tobs,Parama[:,i,:],'--m')
  plt.plot(tobs,parama[:,i],'-m',linewidth=2)
  plt.ylabel('parameter['+str(i)+']')
  plt.xlabel('time')
  plt.grid(True)
 del i 
 plt.subplots_adjust(hspace=0.3)
        

#############################################################################
def plotRH(M,tobs,xt,xpf,rank):
 nbins = M+1
 plt.figure().suptitle('Rank histogram')
 for i in range(3):
  plt.subplot(1,3,i+1)
  plt.hist(rank[:,i],bins=nbins)
  plt.xlabel('x['+str(i)+']')
  plt.axis('tight')
 plt.subplots_adjust(hspace=0.3)
 

#############################################################################
def tileplotlocM(mat, lam,mycmap_out=None,vs_out=None,figout=None):
    if mycmap_out==None:
     mycmap = 'BrBG'
    else:
     mycmap = mycmap_out   
    if vs_out==None:
     vs=[-2,2]   
    else:
     vs = vs_out   
    N1,N2 = np.shape(mat)
    if figout==None:
     plt.figure()
    plt.pcolor(np.array(mat).T,edgecolors='k',cmap=mycmap,vmin=vs[0],vmax=vs[1])
    ymin,ymax = plt.ylim()
    plt.ylim(ymax,ymin)
    #plt.clim(-3,3)
    plt.colorbar()
    plt.title('Location matrix, lambda='+str(lam))
    plt.xlabel('variable number')
    plt.ylabel('variable number')
    plt.xticks(np.arange(0.5,N1+0.5),np.arange(N1))
    plt.yticks(np.arange(0.5,N2+0.5),np.arange(N2))
    
    

#############################################################################
def plotL96_Linerr(lin_error,NLdiff,TLdiff):
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(NLdiff,label='||NL(x+Dx)-NL(x)||')
    plt.plot(TLdiff,label='||TL(Dx)||')
    plt.xlabel('time step')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(lin_error)
    plt.xlabel('time step')
    plt.ylabel('linearisation error')
    plt.tight_layout()
        
#############################################################################
# functions added by G. Hu

def Hov_diagram(x,t,ut):
    cmap=matplotlib.colormaps["RdBu"]
    xmax=np.max(ut)
    xmin=np.min(ut)
    xmax_abs=math.ceil(max(abs(xmax),abs(xmin)))
    if xmax_abs%2==0:
        xmax_abs+=1
    levels=np.linspace(-xmax_abs,xmax_abs,xmax_abs+1)
    plt.contourf(x,t,ut,cmap=cmap,levels=levels,extend='both')
    plt.colorbar()
    plt.xlabel('Model variable',fontsize=14)
    plt.ylabel('Time',fontsize=14)
    plt.title('Hovmöller diagram',fontsize=14)

def timeseries_subplots(Nx,t,ut,ncols,linewidth):
    xmax=np.max(ut)
    xmin=np.min(ut)
    nrows=Nx//ncols
    if Nx%ncols != 0:
        nrows += 1
    fig, axs=plt.subplots(nrows,ncols,sharex=True, sharey=True)
    for irow in range(nrows):
        for icol in range(ncols):
            var=irow*ncols+icol
            if var <= Nx-1:
                axs[irow,icol].plot(t,ut[:,var],'k',linewidth=linewidth)
                axs[irow,icol].grid(True)
                axs[irow,icol].set_ylim(xmin,xmax)
            else:
                axs[irow,icol].axis('off')
    fig.text(0.5, 0.04, 'Time', ha='center')
    fig.text(0.04, 0.5, 'Model variable', va='center', rotation='vertical')
    #fig.suptitle('Time series')   
    return fig, axs

def show_observations(Nx_obs,tobs,yobs,loc_obs,axs,size):
    _,ncols=np.shape(axs)    
    for iobs in range(Nx_obs):
        irow=loc_obs[iobs]//ncols
        icol=loc_obs[iobs]%ncols
        axs[irow,icol].scatter(tobs,yobs[:,iobs],size,'r')

def show_covariances(Bc,title,xlab,ylab,cmap,clim,fontsize):  
    plt.imshow(np.array(Bc),interpolation="nearest",cmap=matplotlib.colormaps[cmap],vmin=clim[0],vmax=clim[1])
    plt.colorbar()
    plt.xlabel(xlab,fontsize=fontsize)
    plt.ylabel(ylab,fontsize=fontsize)
    plt.title(title,fontsize=fontsize)

def add_timeseries(Nx,t,ut,axs,linecolor,linewidth):
    nrows,ncols=np.shape(axs)    
    for irow in range(nrows):
        for icol in range(ncols):
            var=irow*ncols+icol
            if var <= Nx-1:
                axs[irow,icol].plot(t,ut[:,var],linecolor,linewidth=linewidth)
            else:
                axs[irow,icol].axis('off')
                
def calculate_RMSE(Nsteps,ut,datalist,locs):
    rmse=np.zeros((len(datalist),Nsteps,2))
    for i in range(len(datalist)):
        rmse[i,:,0]=rmse_spread(ut[:,locs[0]],datalist[i][:,locs[0]],None,1)
        rmse[i,:,1]=rmse_spread(ut[:,locs[1]],datalist[i][:,locs[1]],None,1)
    return rmse

def compare_RMSE(Nsteps,t,rmse,labels,colors,linewidth,lab_cols):         
    title_txt = ['Observed variables','Unobserved variables']
    fig, axs=plt.subplots(1,2,sharey=True)
    for i in range(2):
        for j in range(len(rmse)):
            axs[i].plot(t,rmse[j,:,i],colors[j],linewidth=linewidth) 
            axs[i].set_title(title_txt[i])            
    axs[0].set_ylabel('RMSE')
    fig.legend(labels=labels, loc="lower center",ncol=lab_cols)
    plt.tight_layout()

def compare_covariances(Bt,Pbt,Lxx,lags,lim,cmap,title):
    fig, axs=plt.subplots(3,lags,sharex=True, sharey=True)
    for icol in range(lags):
        axs[0,icol].imshow(np.array(Bt[:,:,icol]),cmap=matplotlib.colormaps[cmap],vmin=-lim,vmax=lim)  
        axs[1,icol].imshow(np.array(Pbt[:,:,icol]),cmap=matplotlib.colormaps[cmap],vmin=-lim,vmax=lim)  
        axs[2,icol].imshow(np.array(Lxx*Pbt[:,:,icol]),cmap=matplotlib.colormaps[cmap],vmin=-lim,vmax=lim) 
        axs[0,icol].set_title(title+'t='+str(icol))
    axs[0,0].set_ylabel('Exact')
    axs[1,0].set_ylabel('Raw')
    axs[2,0].set_ylabel('Localized') 
