import numpy as np
from sklearn.preprocessing import StandardScaler
#from utils import discretize_uniform

def coral(source, target, fweights=None):
    """
    Correlation alignement 

    Parameters
    ----------
    source: source domain data
    target: target domain data

    Returns
    -------
    source_tsf: transformed source data
    """

    scaler_source, scaler_target = StandardScaler(), StandardScaler()
    source = scaler_source.fit_transform(source)
    target = scaler_target.fit_transform(target)

    Ct = np.cov(target, rowvar=False, fweights=fweights)# + 1 * np.identity(target.shape[1])
    Cs = np.cov(source, rowvar=False)# + 1 * np.identity(source.shape[1])

    Us, s, _ = np.linalg.svd(Cs)
    Ut, t, _ = np.linalg.svd(Ct)

    s = np.diag(s)
    t = np.diag(t)

    inv_s = np.sqrt(np.linalg.pinv(s))
    t = np.sqrt(t)

    whitening = Us.dot(inv_s).dot(Us.T)
    coloring = Ut.dot(t).dot(Ut.T)

    source_tsf = source.dot(whitening).dot(coloring)
    return scaler_target.inverse_transform(source_tsf)
