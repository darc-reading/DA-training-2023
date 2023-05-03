# $RCSfile: rmse_spread.py,v $
# $Revision: 1.2 $
# $Date: 2013/05/19 19:31:31 $
# MATLAB original by Javier Amezcua
# Python conversion by David Livings

import numpy as np

def rmse_spread(xt,xmean,Xens,anawin):
    """Compute RMSE and spread.

    This function computes the RMSE of the background (or analysis) mean with
    respect to the true run, as well as the spread of the background (or
    analysis) ensemble.

    Inputs:  - xt, the true run of the model [length time, N variables]
             - xmean, the background or analysis mean
               [length time, N variables]
             - Xens, the background or analysis ensemble
               [length time, N variables, M ensemble members] or None if
               no ensemble
             - anawin, the analysis window length.  When assimilation
               occurs every time we observe then anawin = period_obs.
    Outputs: - rmse, root mean square error of xmean relative to xt
             - spread, spread of Xens.  Only returned if Xens != None."""

    la,N = xt.shape
    
    # Select only the values at the time of assimilation
    ind = range(0,la,anawin)
    mse = np.mean((xt[ind,:]-xmean[ind,:])**2,axis=1)
    rmse = np.sqrt(mse)
    
    if Xens != None:
        spread = np.var(Xens[ind,:,:],ddof=1,axis=2)
        spread = np.mean(spread,axis=1)
        spread = np.sqrt(spread)
        return rmse,spread
    else:
        return rmse
