# create mSDA representation for amazon review data
import numpy as np
import sys
sys.path.append("./src/")

from io_utils import load_amazon_dataset
from mSDA import msda_fit, msda_forward

for dim in [400, 5000]:
    datalist = []
    for filename in ["books", "dvd", "elec", "kitchen"]:
        for train in [True, False]:
            x, y = load_amazon_dataset(filename, train)
            datalist.append(x)

    x = np.vstack(datalist)
    if dim < 5000:
        x = x[:, :dim]

    _, Wlist = msda_fit(x.T, nb_layers=5)

    for filename in ["books", "dvd", "elec", "kitchen"]:
        for train in [True, False]:
            x, y = load_amazon_dataset(filename, train)
            if dim < 5000:
                x = x[:, :dim]
            x_msda = msda_forward(x.T, Wlist)[:,-dim:]

            if train:
                np.save("./data/preprocessing/{}_msda_{}_train".format(filename, dim), x_msda)
                np.save("./data/preprocessing/{}_label_train".format(filename), y)
            else:
                np.save("./data/preprocessing/{}_msda_{}_test".format(filename, dim), x_msda)
                np.save("./data/preprocessing/{}_label_test".format(filename), y)
