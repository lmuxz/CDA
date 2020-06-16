import numpy as np
import pickle
from sklearn.datasets import load_svmlight_files


def load_amazon_dataset(filename, train=True):
    if train:
        partition = "train"
    else:
        partition = "test"
    x, y = load_svmlight_files(["./data/{}_{}.svmlight".format(filename, partition)])
    x = np.array(x.todense())
    y = np.array((y + 1) / 2, dtype=int)
    return x, y

def load_amazon_msda(filename, train=True, suffix=""):
    if train:
        partition = "train"
    else:
        partition = "test"
    return (np.load("./data/preprocessing/{}_msda{}_{}.npy".format(filename, suffix, partition)), 
            np.load("./data/preprocessing/{}_label_{}.npy".format(filename, partition)))

def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print("Modle Saved")

def load_model(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model
