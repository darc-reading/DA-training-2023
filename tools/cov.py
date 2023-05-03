import numpy as np


def getBcanadian(xt, diff_period, max_var, sample_size):
    """Canadian quick method to obtain the background error covariance
       from model run

    Parameters
    ----------
    xt : ndarry
        model trajectory. shape: (nx, nt)
    diff_period : int
      number of time steps to offset forecast differences
    max_var : float
      The background error variance
    sample_size : int
      Total time steps used for getting covariance matrix

    Returns
    -------
    B : ndarray
        The background covariance matrix
    Bcorr : ndarray
        The correlation matrix
    """
    _, total_steps = xt.shape
    assert total_steps >= sample_size, \
        f'model trajectory length {total_steps} must >= sample_size {sample_size}'

    sam_period = 1
    ind_sample_0 = np.arange(0, sample_size - diff_period, sam_period)
    sam_size = len(ind_sample_0)
    ind_sample_plus = ind_sample_0 + diff_period
    x_sample = xt[:, ind_sample_0] - xt[:, ind_sample_plus]

    Bcorr = np.corrcoef(x_sample)
    B = np.cov(x_sample)

    if max_var:
        alpha = max_var / np.amax(np.diag(B))
        B = alpha * B
    return B, Bcorr


def getBsimple(xt, nt, samfreq=2):
    """A very simple method to obtain the background error covariance.

    Obtained from a long run of a model.

    Parameters
    ----------
    xt : ndarry
        model trajectory. shape: (nx, nt)
    nt : int
        total time steps
    samfreq : int
        sampling frequency of the trajectory. Default: 2

    Returns
    -------
    B : ndarray
        The background covariance matrix
    Bcorr : ndarray
        The correlation matrix
    """
    err2 = 2
    # Precreate the matrix
    ind_sample = range(0, total_steps, samfreq)
    x_sample = xt[:, ind_sample]
    Bcorr = np.corrcoef(x_sample)

    B = np.cov(x_sample)
    alpha = err2/np.amax(np.diag(B))
    B = alpha*B
    return B, Bcorr


if __name__ == "__main__":
    nx = 5
    np.random.seed(100)
    x0 = np.random.random(nx)
    B, Bcorr = getBcanadian(x0, 0.01, 10, None, 8.0, 1000)
    print(B)
    print(Bcorr)
