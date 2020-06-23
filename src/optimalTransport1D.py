import numpy as np
from multiprocessing import Pool


def transport_1d(source, target):
    """
    1D optimal transport with lp norme
        - sort source and target data
        - fill the target warehouse by source stock
    
    Parameters
    ----------
    source: numerical encoding for source domain
    target: numerical encoding for target domain
    """
    res = np.zeros_like(source).astype(float)
    order = np.argsort(target)

    target_n = target.shape[0]

    source_value = source
    source = np.argsort(source)
    source_n = source.shape[0]

    cur_value = 0
    wdistance = 0 # need to change 
    cur_capacity = 1/target_n # need to change
    for p in source:
        stock = 1/source_n
        value = 0
        while stock > 0:
            if stock >= cur_capacity:
                stock -= cur_capacity
                value += target[order[cur_value]] * cur_capacity * source_n
                wdistance += abs(target[order[cur_value]] - source_value[p]) * cur_capacity * source_n
                cur_value = min([cur_value+1, target_n-1])
                cur_capacity = 1/target_n
            else:
                cur_capacity -= stock
                value += target[order[cur_value]] * stock * source_n
                wdistance += abs(target[order[cur_value]] - source_value[p]) * stock * source_n
                stock = 0
        res[p] = value
    return res, wdistance


def transport_1d_helper(args):
    return transport_1d(*args)


class optimalTransport1D:
    """
    optimal transport 1d by 1d and point to point in multi-thread
    """
    def __init__(self):
        pass

    def fit_transform(self, source, target, njobs=1):
        """
        Find the best barycenter for optimal transport

        Parameters
        ----------
        source: source data
        target: target data
        njobs: number of threads

        Returns
        -------
        source_ot: transformed source data
        """
        args = []
        for i in range(source.shape[-1]):
            args.append([source[~np.isnan(source[:,i]),i].copy(), target[~np.isnan(target[:,i]),i].copy()])

        with Pool(njobs) as p:
            res = p.map(transport_1d_helper, args)
        
        res, wdistance = zip(*res)
        X_ot = source.copy()
        for i in range(X_ot.shape[-1]):
            X_ot[~np.isnan(X_ot[:,i]),i] = res[i]

        return X_ot, np.array(wdistance)
