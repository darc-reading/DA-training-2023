"""
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


def plotL96(t, xt, nx, title):
    """plot L96

    Parameters
    ----------
    t : ndarray
        1D array of model time
    xt : ndarry
        All time snapshots of the model state. shape: (nx, nt)
    nx : int
        dimension of model state
    title: : str
        title of the plot
    """
    plt.figure()
    largest = max([np.abs(np.min(xt)), np.abs(np.max(xt))])
    levs = np.linspace(-largest,largest,21)
    mycmap = plt.get_cmap("BrBG", 21)
    C = plt.contourf(np.arange(nx), t, xt.T, cmap=mycmap, levels=levs)
    plt.contour(np.arange(nx), t, xt.T, 10, colors="k", linestyles="solid")
    plt.xlabel("variable number")
    plt.ylabel("time")
    plt.title("Hovmoller diagram: " + title)
    plt.colorbar(C, extend="both")


def plotL96_Linerr(lin_error, NLdiff, TLdiff):
    """plot linear error of L96

    Parameters
    ----------
    NLdiff : ndarray
        1D array of ||NL(x+Dx)-NL(x)||
    TLdiff : ndarry
        1D array of ||TL(Dx)||
    lin_error : int
        Difference between NLdiff and TLdiff
    """
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(NLdiff, label="||NL(x+Dx)-NL(x)||")
    plt.plot(TLdiff, label="||TL(Dx)||")
    plt.xlabel("time step")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(lin_error)
    plt.xlabel("time step")
    plt.ylabel("linearisation error")
    plt.tight_layout()


def tileplotB(mat, mycmap_out=None, vs_out=None, figout=None):
    """plot background error covariance

    Parameters
    ----------
    mat : ndarray
        The background error covariance matrix
    mycmap_out : str or matplotlib.colors.Colormap
        plotting colormap
    vs_out : tuple or list
        vmin and vmax of the plot
    figout : matplotlib.figure
        The figure object from matplotlib
    """
    if mycmap_out == None:
        mycmap = "BrBG"
    else:
        mycmap = mycmap_out
    if vs_out == None:
        vs = [-2, 2]
    else:
        vs = vs_out
    N1, N2 = np.shape(mat)
    if figout == None:
        plt.figure()
    plt.pcolor(np.array(mat).T, edgecolors="k", cmap=mycmap, vmin=vs[0], vmax=vs[1])
    ymin, ymax = plt.ylim()
    plt.ylim(ymax, ymin)
    plt.colorbar(extend="both")
    plt.title("matrix")
    plt.xlabel("variable number")
    plt.ylabel("variable number")
    plt.xticks(np.arange(0.5, N1 + 0.5), np.arange(N1))
    plt.yticks(np.arange(0.5, N2 + 0.5), np.arange(N2))


def plotH(H):
    """Plot observation operator
    Parameters
    ----------
    H : ndarray
        The observation operator matrix
    """
    ny, nx = H.shape

    extent = [
        0.5,
        float(nx) + 0.5,
        float(ny) + 0.5,
        0.5,
    ]  # To set axis label limits, otherwise just labels pixels

    fig, ax = plt.subplots()
    cax = ax.imshow(H, interpolation="None", cmap="YlOrBr", extent=extent)
    cbar = fig.colorbar(cax, orientation="horizontal")
    ax.set_title("Observation operator, H")
    ax.set_xlabel("state element", fontsize=16)
    ax.set_ylabel("ob number", fontsize=16)


def plotL96obs(tobs, y, ny, title):
    """Plot observations
    Parameters
    ----------
    tobs : ndarray
        The one dimensional time array of the observations
    y : ndarray
        The observations, shape: (ny, nobt)
    ny : int
        The number of observations
    title: : str
        title of the plot
    """
    levs = np.linspace(-15, 15, 21)
    mycmap = plt.get_cmap("BrBG", 21)
    plt.figure().suptitle(title)
    y_trans = np.transpose(y)
    C = plt.contourf(np.arange(ny), tobs, y_trans, cmap=mycmap, levels=levs)
    plt.contour(np.arange(ny), tobs, y_trans, colors="k", linestyles="solid")
    plt.xlabel("ob number")
    plt.ylabel("ob time")
    plt.title(title)
    plt.colorbar(C, extend="both")


def plotRMSP(exp_title,t,rmseb=None,rmsea=None,spreadb=None,spreada=None):
    """Plot root mean squared error and spread of ensemble
    Parameters
    ----------
    exp_title : str
        figure title
    t : ndarray
        time array. shape: nt
    rmseb : ndarray
        RMSE of background. shape: nt
    rmsea : 
        RMSE of analysis. shape: nt
    spreadb : 
        Ensemble spread of background. shape: nt
    spreada : 
        Ensemble spread of analysis. shape: nt
    """
    plt.figure()
    plt.subplot(2,1,1)
    if np.all(rmseb)!=None:
        plt.plot(t, rmseb,'b',label='background')
    plt.plot(t, rmsea,'m',label='analysis')
    plt.legend()
    plt.ylabel('RMSE')
    plt.xlabel('time')
    plt.title(exp_title)
    plt.grid(True)

    if np.all(spreadb)!=None:
        plt.subplot(2,2,3)
        if np.all(rmseb)!=None:
            plt.plot(t,rmseb,'b',label='RMSE')
        plt.plot(t,spreadb,'--k',label='spread')
        plt.legend()
        plt.title('background')
        plt.xlabel('time')
        plt.grid(True)

    if np.all(spreada)!=None:
        plt.subplot(2,2,4)
        plt.plot(t,rmsea,'m',label='RMSE')
        plt.plot(t,spreada,'--k',label='spread')
        plt.legend()
        plt.title('analysis')
        plt.xlabel('time')
        plt.grid(True)

    plt.subplots_adjust(hspace=0.25)

  
def plotDA_kf(t,xt,tobs,H,y,Xb,xb,Xa,xa,exp_title):
    """plot the KF results in L96
    Note: it does not work with direct observation on grid points
    e.g. it doesn't work with foot_6 or foot_cent.
    It might be better to plot in obs. space instead of 
    model space (todo?) !!!

    Parameters
    ----------
    t : ndarray
        1D array of model time. shape: nt
    xt : ndarry
        model truth. shape: (nx, nt)
    tobs : ndarray
        observation time. shape: nobt
    H : ndarray
        observation operator matrix. shape: (ny, nx)
    y : ndarray
        observations. shape: (ny, nobt)
    Xb: ndarray
        the background ensemble 3D array. shape: [nx, ne, nt]
    xb: ndarray
        background mean. shape: [nx, nt]
    Xa: ndarray
        the analysis ensemble 3D array. shape: [nx, ne, nt]
    xa: ndarray
        analysis mean. shape: [nx, nt]
    exp_title: : str
        title of the plot
    """
    assert np.sum(H) == len(H), \
        'Make sure H only has 0 and 1 entries for direct grid points observations'
    nx, _ = xt.shape
    plt.figure().suptitle('Ensemble:'+exp_title)
    for i in range(nx):
        plt.subplot(int(np.ceil(nx/4)), 4, i+1)
        plt.plot(t, xt[i], 'k')
        plt.plot(t, Xb[i].T, '--b')
        plt.plot(t, Xa[i].T, '--m')
        if np.sum(H[:, i]) > 0:
            # prevent scatter() from rescaling axes
            # plt.autoscale(False)
            plt.scatter(tobs, y[H[:, i] > 0.], 20, 'r')
        plt.ylabel('x['+str(i)+']')
        plt.xlabel('time')
        plt.grid(True)

    plt.subplots_adjust(wspace=0.7,hspace=0.3)

    plt.figure().suptitle(exp_title)
    for i in range(int(nx)):
        plt.subplot(int(np.ceil(nx/4.0)),4,i+1)
        plt.plot(t,xt[i],'k',label='truth')
        if np.sum(H[:, i]) > 0:
            # prevent scatter() from rescaling axes
            # plt.autoscale(False)
            plt.scatter(tobs,y[H[:, i] > 0.], 20, 'r', label='obs')
        plt.plot(t,xb[i],'b',label='background')
        plt.plot(t,xa[i],'m',label='analysis')
        plt.ylabel('x['+str(i)+']')
        plt.xlabel('time')
        plt.grid(True)
    plt.legend()
    plt.subplots_adjust(wspace=0.7,hspace=0.3)


def plotRH(ne,rank):
    """plot rank histogram for the first 3 variables

    Parameters
    ----------
    ne : int
        ensemble size
    rank : ndarray
        rank of the truth in the ensemble for all time steps. shape: nt
    """
    nbins = n+1
    plt.figure().suptitle('Rank histogram')
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.hist(rank[:,i],bins=nbins)
        plt.xlabel('x['+str(i)+']')
        plt.axis('tight')
    plt.subplots_adjust(hspace=0.3)


def tileplotlocM(mat, lam, mycmap_out=None,vs_out=None,figout=None):
    """plot localisation matrix

    Parameters
    ----------
    mat : ndarray
        The background error covariance matrix
    lam : float or int
        localisation radius
    mycmap_out : str or matplotlib.colors.Colormap
        plotting colormap
    vs_out : tuple or list
        vmin and vmax of the plot
    figout : matplotlib.figure
        The figure object from matplotlib
    """
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


def plot_matrix (mat, title, limits=None):
    """plot matrix

    Parameters
    ----------
    mat : ndarray
        The matrix to be plotted
    title : str
        plot title
    limits : float
        The maximum limit of the colourbar
    """
    if limits == None:
        # No plotting limits specified, so find limits
        lims   = [abs(np.amin(mat)), abs(np.amax(mat))]
        minmax = max(lims)
    else:
        minmax = limits

    n1, n2 = np.shape(mat)
    C = plt.pcolor(np.array(mat), edgecolors='k', cmap='BrBG', vmin=-1.0*minmax, vmax=minmax)

    plt.colorbar(extend='both')
    plt.title(title)
    plt.xlabel('variable number')
    plt.ylabel('variable number')
    plt.xticks(np.arange(0.5, n1+0.5), np.arange(n1))
    plt.yticks(np.arange(0.5, n2+0.5), np.arange(n2))


def plot_LocMatrix(P, name='matrix'):
    """plot localisation matrix

    Parameters
    ----------
    P : ndarray
        The matrix to be plotted
    name : str
        plot title. Default: matrix
    """
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


def plotL96_obs_vs_modelobs(x_traj, y_traj, H, period_obs, exp_title):
    """Compare observation and model state in obs. space

    Parameters
    ----------
    x_traj : ndarray
        model trajectory. shape: (nx, nt)
    y_traj : ndarry
        observation trajecty. shape: (ny, nobt)
    H : ndarray
        observation operator matrix. shape: (ny, nx)
    period_obs : int
        number of time steps between observations
    exp_title: : str
        title of the plot
    """
    ny, nobstimes = y_traj.shape

    # Set-up the arrays used to store the observations and model observations
    obs = y_traj.ravel(order='F')
    model_obs = np.concatenate([H@x_traj[:, obtime*period_obs].ravel()
                                for obtime in range(nobstimes)
                                ])

    # Plot the obs vs model obs
    plt.figure().suptitle(exp_title)
    plt.scatter(model_obs[:], obs[:])
    plt.xlabel("model observations")
    plt.ylabel("observations")
    plt.grid(True)


def make_histogram(nbins, xtrue_traj, x_traj, model_times, anal_times, title):
    """Plot histogram of values minus truth

    Parameters
    ----------
    nbins : int
        Number of bins for hte histogram
    xtrue_traj : ndarray
        truth model trajectory. shape: (nx, nt)
    x_traj : ndarry
        model trajectory. shape: (nx, nt)
    model_times : ndarray
        array of model times. shape: (nt)
    anal_times : ndarray
        array of analysis times. shape: (nobt)
    title: : str
        title of the plot
    """
    nx, nt = xtrue_traj.shape
    # Make flattened arrays of truth and analyses at analysis times
    # one-liner based on
    # https://stackoverflow.com/questions/58623100/float-rounding-error-with-numpy-isin-function
    selectAnaTime = np.isclose(anal_times[:,None],
                               model_times, atol=1e-7).any(axis=0)
    truth = xtrue_traj[:, selectAnaTime].ravel(order='F')
    x = x_traj[:, selectAnaTime].ravel(order='F')

    Npopulation_t = len(truth)
    Npopulation_e = len(x)

    assert Npopulation_e == Npopulation_t, \
        f"the populations of the truth ({Npopulation_t}) " \
        "and analysis ({Npopulation_e}) are meant to be the same."

    # Compute the error
    error = x - truth

    # Compute the mean and standard deviation
    mean = np.mean(error)
    std = np.std(error)

    # Find the maximum absolute value of these differences
    maxval = max(abs(error))

    # Generate the bins
    bins = np.linspace(-1.0 * maxval, maxval, nbins + 1)

    # Plot histogram
    fig, ax = plt.subplots()
    ax.hist(error, bins=bins, histtype="bar", facecolor="b")
    ax.set_title(
        title
        + "\nmean = "
        + str.format("%.5f" % mean)
        + ", stddev = "
        + str.format("%.5f" % std)
        + ", population = "
        + str(Npopulation_t)
    )
    ax.set_xlabel("estimate - truth")
    ax.set_ylabel("frequency")


def plot_log_test(x, y, xaxname, yaxname, title):
    """Plot correctness diagram or gradient test
    
    This can be used to test correctness of the TL model

    Parameters
    ----------
    x : ndarray
        multiple of perturbation
    y : ndarray
        relative error
    xaxname : str
        x-axis label
    yaxname : str
        y-axis label
    title: : str
        title of the plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, "k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.title(title)
    plt.xlabel(xaxname)
    plt.ylabel(yaxname)
    plt.grid(True)


def plotL96DA_var(
    t, xt_traj, xb_traj, xa_traj, tobs, y_traj, anal_times, H, R, exp_title
):
    """Plot L96 results for variational method

    Parameters
    ----------
    t : ndarray
        1D array of model time. shape: nt
    xt_traj : ndarray
        truth model trajectory. shape: (nx, nt)
    xb_traj : ndarry
        background model trajecty. shape: (ny, nt)
    xa_traj : ndarray
        analysis model trajectory. shape: (nx, nt)
    tobs : ndarray
        observation time. shape: nobt
    y_traj : ndarray
        observation trajectory. shape: (nx, nobt)
    anal_times : ndarray
        array of analysis times. shape: (nobt)
    H : ndarray
        observation operator matrix. shape: (ny, nx)
    period_obs : int
        number of time steps between observations
    R : ndarray
        observation error covariance matrix. shape: (ny, ny)
    exp_title: : str
        title of the plot
    """
    nx, _ = xt_traj.shape
    ny, _ = y_traj.shape

    for i in range(nx):
        plt.figure().suptitle(exp_title + " variable " + str(i))
        # ax = plt.subplot(n, 1, i+1)

        # Plot vertical lines at analysis points
        for anal_time in anal_times:
            plt.axvline(x=anal_time, color="y")

        plt.plot(t, xt_traj[i], "k", label="truth")
        plt.plot(t, xb_traj[i], "b", label="background")
        plt.plot(t, xa_traj[i], "r", label="analysis")

        # Is this state component directly observed?
        idx = np.arange(ny)[H[:, i] == 1.0]
        ob_component_no = np.max(idx) if len(idx) > 0 else -1

        if ob_component_no > -1:
            plt.errorbar(
                tobs[:],
                y_traj[ob_component_no, :],
                yerr=np.sqrt(R[ob_component_no, ob_component_no]),
                color="g",
                ls="none",
            )

        plt.ylabel("x[" + str(i) + "]")
        plt.xlabel("time")
        plt.legend()
        plt.grid(True)


def plotL63(t, xt):
    """plot L63

    Parameters
    ----------
    t : ndarray
        1D array of model time
    xt : ndarry
        All time snapshots of the model state. shape: (nx, nt)
    """
    plt.figure().suptitle('Lorenz 63 system - Truth')
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(t,xt[i],'k')
        plt.ylabel('x['+str(i)+']')
        plt.xlabel('time')
        plt.grid(True)
        plt.subplots_adjust(hspace=0.3)

    fig = plt.figure()
    fig.suptitle('Lorenz 63 system - Truth')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xt[0],xt[1],xt[2],'k')
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_zlabel('x[2]')
    ax.grid(True)


def plotL63obs(t,xt,tobs,H,y,exp_title):
    """plot the obs. along with truth in L63
    Note: it does not work with direct observation on grid points
    e.g. it doesn't work with foot_6 or foot_cent.
    It might be better to plot in obs. space instead of 
    model space (todo?) !!!

    Parameters
    ----------
    t : ndarray
        1D array of model time. shape: nt
    xt : ndarry
        model truth. shape: (nx, nt)
    tobs : ndarray
        observation time. shape: nobt
    H : ndarray
        observation operator matrix. shape: (ny, nx)
    y : ndarray
        observations. shape: (ny, nobt)
    exp_title: : str
        title of the plot
    """
    plt.figure().suptitle(exp_title)
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(t,xt[i],'k',label='truth')
        if np.sum(H[:, i]) > 0:
            # plt.autoscale(False) # prevent scatter() from rescaling axes
            plt.scatter(tobs,y[H[:, i]>0],20,'r', label='obs')
    plt.ylabel('x['+str(i)+']')
    plt.xlabel('time')
    plt.grid(True)
    plt.subplots_adjust(hspace=0.3)


def plotpar(Nparam,tobs,paramt_time,Parama,parama):
    """plot the model parameters

    Parameters
    ----------
    Nparam : int
        number of model parameters
    tobs : ndarray
        1D array of model time. shape: nobt
    paramt_time : ndarray
        Truth of parameter. shape: (Nparam, nobt)
    Parama : ndarray
        Ensemble analyais of parameter. shape: (Nparam, ne, nobt)
    parama : ndarray
        Ensemble averaged analysis of parameter. shape: (Nparam, nobt)
    """
    plt.figure().suptitle('True Parameters and Estimated Parameters')
    for i in range(Nparam):
        plt.subplot(Nparam,1,i+1)
        plt.plot(tobs,paramt_time[i],'k')
        plt.plot(tobs,Parama[i].T,'--m')
        plt.plot(tobs,parama[i],'-m',linewidth=2)
        plt.ylabel('parameter['+str(i)+']')
        plt.xlabel('time')
        plt.grid(True)

    plt.subplots_adjust(hspace=0.3)


def plotL63DA_kf(t,xt,tobs,H,y,Xb,xb,Xa,xa,exp_title):
    """Plot L96 results for variational method

    Parameters
    ----------
    t : ndarray
        1D array of model time. shape: nt
    xt : ndarry
        model truth. shape: (nx, nt)
    tobs : ndarray
        observation time. shape: nobt
    H : ndarray
        observation operator matrix. shape: (ny, nx)
    y : ndarray
        observations. shape: (ny, nobt)
    Xb: ndarray
        the background ensemble 3D array. shape: [nx, ne, nt]
    xb: ndarray
        background mean. shape: [nx, nt]
    Xa: ndarray
        the analysis ensemble 3D array. shape: [nx, ne, nt]
    xa: ndarray
        analysis mean. shape: [nx, nt]
    exp_title: : str
        title of the plot
    """
    #plt.figure().suptitle('Truth, Observations, Background Ensemble and Analysis Ensemble')
    plt.figure().suptitle('Ensemble:'+exp_title)
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(t,xt[i],'k',label='truth')
        plt.plot(t,Xb[i].T,'--b',label='background')
        plt.plot(t,Xa[i].T,'--m',label='analysis')
        if np.sum(H[:, i]) > 0:
        #   plt.autoscale(False) # prevent scatter() from rescaling axes
            plt.scatter(tobs,y[H[:, i] > 0.],20,'r',label='obs')
        plt.ylabel('x['+str(i)+']')
        plt.xlabel('time')
        plt.grid(True)
        plt.subplots_adjust(hspace=0.3)

 
    #plt.figure().suptitle('Truth, Observations, Background and Analysis Mean')
    plt.figure().suptitle(exp_title)
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(t,xt[i],'k',label='truth')
        plt.plot(t,xb[i],'b',label='background')
        plt.plot(t,xa[i],'m',label='analysis')
        if np.sum(H[:, i]) > 0:
        #   plt.autoscale(False) # prevent scatter() from rescaling axes
            plt.scatter(tobs,y[H[:, i] > 0.],20,'r',label='obs')
        plt.ylabel('x['+str(i)+']')
        plt.xlabel('time')
        plt.grid(True)
    plt.subplots_adjust(hspace=0.3)
    plt.legend()
