import numpy as np
from multiprocessing import Pool


def transport_1d(source, target, target_remap=None, cate=False):
    """
    1D optimal transport with lp norme
        - sort source and target data
        - fill the target warehouse by source stock
    
    Parameters
    ----------
    source: numerical encoding for source domain (if target_remap is None)
    target: numerical encoding for target domain (if target_remap is None)
    target_remap: the real value of target
    cate: if the value of target_remap is categorical or not 
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
    for i, p in enumerate(source):
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
        if cate:
            res[p] = int(round(value))
        else:
            res[p] = value
    return res, wdistance


def transport_1d_helper(args):
    return transport_1d(*args)


def landmark_optimal_transport(source, target, target_label, landmark, eta, k=10):
    landmark = sorted(landmark)
    percentile = []
    for m in landmark:
        percentile.append((source < m).sum() / source.shape[0] * 100)

    exclude = []
    i = 0
    while i < len(percentile):
        if percentile[i] <= eta:
            exclude.append(i)
            i += 1
        elif i < len(percentile) - 1 and percentile[i+1] <= percentile[i] + eta:
            exclude.append(i+1)
            i += 2
        else:
            i += 1
    percentile = [percentile[i] for i in range(len(percentile)) if i not in exclude]
    landmark = [landmark[i] for i in range(len(landmark)) if i not in exclude]

    target_mark = [] # computed using percentile
    for p in percentile:
        target_mark.append([np.percentile(target, p-eta, interpolation="nearest"), 
            np.percentile(target, p, interpolation="nearest"), 
            np.percentile(target, p+eta, interpolation="nearest")])
    
    target_landmark = [] # mapping with source landmark
    for m1, m, m2 in target_mark:
        index = (target>=(m1)) & (target<(m2))
        sub_label = target_label[index]
        sub_target = target[index]

        index = np.argsort(sub_target)
        sub_target = sub_target[index]
        sub_label = sub_label[index] # sorted label

        max_index = np.where(sub_target == m)[0][0]
        max_value = sub_label[ max([max_index-int(k/2), 0]) : max_index+int(k/2) ].sum()
        for i in range(sub_label.shape[0]):
            knn_fraud = sub_label[ max([i-int(k/2), 0]) : i+int(k/2) ].sum()
            if knn_fraud > max_value:
                max_value = knn_fraud
                max_index = i
        
        target_landmark.append(sub_target[max_index])
    
    return target_landmark


class optimalTransport1D:
    """
    optimal transport 1d by 1d and point to point in multi-thread
    """
    def __init__(self):
        pass

    def fit_transform(self, source, target, njobs=1, target_remap=None, cates=[]):
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
        if target_remap is not None:
            for i in range(source.shape[-1]):
                if i in cates:
                    args.append([source[~np.isnan(source[:,i]),i].copy(), 
                                target[~np.isnan(target[:,i]),i].copy(), 
                                target_remap[~np.isnan(target[:,i]),i].copy(), True])
                else:
                    args.append([source[~np.isnan(source[:,i]),i].copy(), 
                                target[~np.isnan(target[:,i]),i].copy(), 
                                target_remap[~np.isnan(target[:,i]),i].copy(), False])
        else:
            for i in range(source.shape[-1]):
                args.append([source[~np.isnan(source[:,i]),i].copy(), target[~np.isnan(target[:,i]),i].copy()])

        with Pool(njobs) as p:
            res = p.map(transport_1d_helper, args)
        
        res, wdistance = zip(*res)
        X_ot = source.copy()
        for i in range(X_ot.shape[-1]):
            X_ot[~np.isnan(X_ot[:,i]),i] = res[i]

        return X_ot, np.array(wdistance)
