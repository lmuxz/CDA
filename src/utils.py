import numpy as np
from sklearn.metrics import auc, precision_recall_curve

def reduce_dataset(xs, xt):
    n = xt.shape[0]
    indext = np.random.choice(range(xs.shape[0]), n, replace=True)
    return indext

def extend_dataset(xs, xt):
    ns = xs.shape[0]
    nt = xt.shape[0]
    n = max(ns, nt)
    indexs = np.random.choice(range(ns), n, replace=True)
    indext = np.random.choice(range(nt), n, replace=True)
    return_s = list(range(ns)) + list(indexs[ns:n])
    return_t = list(range(nt)) + list(indext[nt:n])
    return return_s, return_t

def pr_auc(pred, label):
    p, r, _ = precision_recall_curve(label, pred)
    order = np.argsort(r)
    return auc(r[order], p[order])

def similarity_to_dissimilarity(sim):
    max_sim = sim.max()
    return max_sim - sim

def of_uni_cate(cate):
    modality, counts = np.unique(cate, return_counts=True)
    density = counts / counts.sum()

    res = 1 / (1+np.log(counts.sum()/counts).reshape((-1, 1)).dot(np.log(counts.sum()/counts).reshape(1, -1)))
    identity = modality.repeat(len(modality)).reshape((len(modality), len(modality))) == modality
    res[identity] = 1

    return res, modality

def performance(pred, label, auc=True, first_n=None):
    if first_n is not None:
        index = np.argsort(pred)[-first_n:]
        pred = pred[index]
        label = label[index]
    if auc:
        return pr_auc(pred, label)
    else:
        return label.sum()


def best_prediction(model, data, data_tsf, feature_list, best_index, repeat=1):
    """
    Inputs:
        model: prediction model, should have predict_prob function
        data: always the same size as data_tsf
        data_tsf: 
        feature_list:
        best_index: when to stop, include
        repeat: Default 1
    Returns:
        Best prediction
    """
    data = data.copy()
    for f in feature_list[:best_index+1]:
        for ff in f:
            data[:, ff] = data_tsf[:, ff]
    pred = model.predict_prob(data)
    pred = pred.reshape(repeat, -1).mean(axis=0)
    return pred


def bootstrap(pred, label, n_sample, index_list=None):
    """
    Inputs:
        pred: 1d probability list
        label: real label of transaction
        n_sample: number of repeation
    Returns:
        perf_dist: the experimental distribution of performance in pr-auc
    """
    
    perf_dist = []
    if index_list is None:
        index_list_res = []
    else:
        index_list_res = index_list
    for i in range(n_sample):
        if index_list is None:
            index = np.random.choice(range(len(pred)), len(pred), replace=True)
            index_list_res.append(index)
        else:
            index = index_list[i]
        pr_auc = performance(pred[index], label[index])
        perf_dist.append(pr_auc)
    
    return np.array(perf_dist), index_list_res

def test_interval(perf_dist, perf_dist_old):
    """
    Inputs:
        perf_dist: new perf_dist
        perf_dist_old: old perf_dist
    Returns:
        sum of probability
    """
    prob = (np.array(perf_dist) > np.array(perf_dist_old)).sum() / len(perf_dist_old)
    #print("Test Interval", prob)
    return prob


def greedy_search(clf, xt, target_ot, yt, explore_features=[], n_sample=100, verbose=True):
    xt = xt.copy()
    index_list = None
    feature_list = []
    
    perf_list = []
    pred_list = []
    perf_dist_list = []

    #n = target_ot.shape[1]
    pred = clf.predict_prob(xt)
    pred_list.append(pred)

    perf = performance(pred, yt)
    perf_list.append(perf)
    
    perf_dist, index_list = bootstrap(pred, yt, n_sample=n_sample, index_list=index_list)
    perf_dist_list.append(perf_dist)
    if verbose:
        print(perf)
    
    early_stop = False
    while (len(feature_list) < len(explore_features)) and not early_stop:
        best_feature = None
        best_pred = None
        best_perf = -float('inf')
        # Find the best feature by greedy search
        for i in explore_features:
            if i not in feature_list:
                xt_ot = xt.copy()
                
                for f in i:
                    xt_ot[:, f] = target_ot[:, f]
                
                pred = clf.predict_prob(xt_ot)
                perf = performance(pred, yt)
                if perf > best_perf:
                    best_feature = i
                    best_pred = pred
                    best_perf = perf

        perf_dist, index_list = bootstrap(best_pred, yt, n_sample=n_sample, index_list=index_list)
        early_stop = test_interval(perf_dist, perf_dist_list[-1]) < 0.5
        
        if not early_stop:
            for f in best_feature:
                xt[:, f] = target_ot[:, f]
            feature_list.append(best_feature)
            perf_list.append(best_perf)
            pred_list.append(best_pred)
            perf_dist_list.append(perf_dist)

        if verbose:
            print(best_perf)

    return perf_list, feature_list, pred_list, perf_dist_list, index_list


def greedy_search_cate(clf, xt, target_ot, yt, explore_features=[], repeat=10, n_sample=100, verbose=True):
    xt = xt.copy()
    index_list = None
    feature_list = []
    
    perf_list = []
    pred_list = []
    perf_dist_list = []

    #n = target_ot.shape[1]
    pred = clf.predict_prob(xt)
    pred = pred.reshape(repeat, -1).mean(axis=0)
    pred_list.append(pred)

    perf = performance(pred, yt)
    perf_list.append(perf)
    
    perf_dist, index_list = bootstrap(pred, yt, n_sample=n_sample, index_list=index_list)
    perf_dist_list.append(perf_dist)
    if verbose:
        print(perf)
    
    early_stop = False
    while (len(feature_list) < len(explore_features)) and not early_stop:
        best_feature = None
        best_pred = None
        best_perf = -float('inf')
        # Find the best feature by greedy search
        for i in explore_features:
            if i not in feature_list:
                xt_ot = xt.copy()
                
                for f in i:
                    xt_ot[:, f] = target_ot[:, f]
                
                pred = clf.predict_prob(xt_ot)
                pred = pred.reshape(repeat, -1).mean(axis=0)
                perf = performance(pred, yt)
                if perf > best_perf:
                    best_feature = i
                    best_pred = pred
                    best_perf = perf

        perf_dist, index_list = bootstrap(best_pred, yt, n_sample=n_sample, index_list=index_list)
        early_stop = test_interval(perf_dist, perf_dist_list[-1]) < 0.5
        
        if not early_stop:
            for f in best_feature:
                xt[:, f] = target_ot[:, f]
            feature_list.append(best_feature)
            perf_list.append(best_perf)
            pred_list.append(best_pred)
            perf_dist_list.append(perf_dist)

        if verbose:
            print(best_perf)

    return perf_list, feature_list, pred_list, perf_dist_list, index_list