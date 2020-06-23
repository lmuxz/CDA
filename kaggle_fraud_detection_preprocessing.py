import sys
sys.path.append("./src/")

import numpy as np
import pandas as pd
import torch
import pickle

from sklearn.preprocessing import LabelEncoder

from nnModel import FC_embedding, EmbeddingModel

from utils import *
from io_utils import *

device = torch.device("cuda")

trans = pd.read_csv("./data/train_transaction.csv")
identity = pd.read_csv("./data/train_identity.csv")
trans = trans.merge(identity[["TransactionID", "DeviceType"]], on="TransactionID")

source = trans[trans.DeviceType=="mobile"]
target = trans[trans.DeviceType=="desktop"]

nan_percent = trans.isnull().sum(axis=0) / trans.shape[0]

ignore = nan_percent[nan_percent > 0.01].index.values.tolist() + ["isFraud", "TransactionDT", "DeviceType"]

source_label = source["isFraud"].values
target_label = target["isFraud"].values

source = source[[f for f in source.columns if f not in ignore]]
target = target[[f for f in target.columns if f not in ignore]]

source = pd.merge(identity, source, how="right", on="TransactionID")
target = pd.merge(identity, target, how="right", on="TransactionID")

nan_percent = source.append(target).isnull().sum(axis=0) / source.append(target).shape[0]

ignore = nan_percent[nan_percent > 0.01].index.values.tolist() + ["TransactionID", "DeviceType"]

source = source[[f for f in source.columns if f not in ignore]]
target = target[[f for f in target.columns if f not in ignore]]

source_index = np.where(~np.any(source.isnull().values, axis=1))[0]
target_index = np.where(~np.any(target.isnull().values, axis=1))[0]

cates = ["id_12", "id_15", "id_28", "id_29", "id_31", "id_35", "id_36", "id_37", "id_38", 
         "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6"]
no_cates = [c for c in source.columns if c not in cates]
source = source[cates+no_cates]
target = target[cates+no_cates]

for c in cates:
    encoder = LabelEncoder()
    encoder.fit(source[c].append(target[c]).astype(str))
    source[c] = encoder.transform(source[c].astype(str))
    target[c] = encoder.transform(target[c].astype(str))
    
cates = ["id_15", "id_31","ProductCD", "card2", "card3", "card4", "card5", "card6"]
no_cates = [c for c in source.columns if c not in cates]
source = source[cates+no_cates]
source.drop("card1", inplace=True, axis=1)
target = target[cates+no_cates]
target.drop("card1", inplace=True, axis=1)

source = source.values[source_index]
target = target.values[target_index]

source_label = source_label[source_index]
target_label = target_label[target_index]


min_values = np.min(np.r_[source, target], axis=0)

source = source - min_values
target = target - min_values

for i in range(8, 120):
    source[:,i] = np.log(1 + source[:,i])
    target[:,i] = np.log(1 + target[:,i])
    
np.random.seed(12345)
torch.manual_seed(12345)

embd = FC_embedding()

params = {
    "epoch": 50,
    "batch_size": 128,
    "learning_rate": 0.001,
    "pos_weight": 1,
    "model": embd,
    "device": device,
}
model = EmbeddingModel(**params)

source_size = int(source.shape[0] / 4)

model.fit(source[:source_size*3], source_label[:source_size*3], 
          source[source_size*3:], source_label[source_size*3:], verbose=True)

embedding_dict = [model.model.embed[i].state_dict()["weight"].detach().cpu().numpy() for i in range(8)]
with open("./data/embedding_dict_kaggle2.pkl", "wb") as file:
    pickle.dump(embedding_dict, file)

np.save("./data/mobile_trans", source)
np.save("./data/mobile_label", source_label)
np.save("./data/desktop_trans", target)
np.save("./data/desktop_label", target_label)