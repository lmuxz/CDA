import torch
import pickle
import numpy as np
import progressbar as pb

from torch import nn
from torch import optim
from torch.utils import data


class FC_embedding(nn.Module):
    def __init__(self, embedding_dict=[], embedding_file=None):
        super().__init__()
        #embedding_num = [704, 134, 702, 704, 11, 220, 3, 219]
        #embedding_dim = [5, 3, 5, 5, 2, 4, 1, 4]
        embedding_num = [3, 131, 4, 483, 103, 5, 106, 4]
        embedding_dim = [1, 3, 1, 4, 3, 1, 3, 1]

        self.embed0 = nn.Embedding(embedding_num[0], embedding_dim[0])
        self.embed1 = nn.Embedding(embedding_num[1], embedding_dim[1])
        self.embed2 = nn.Embedding(embedding_num[2], embedding_dim[2])
        self.embed3 = nn.Embedding(embedding_num[3], embedding_dim[3])
        self.embed4 = nn.Embedding(embedding_num[4], embedding_dim[4])
        self.embed5 = nn.Embedding(embedding_num[5], embedding_dim[5])
        self.embed6 = nn.Embedding(embedding_num[6], embedding_dim[6])
        self.embed7 = nn.Embedding(embedding_num[7], embedding_dim[7])

        self.embed = [self.embed0, 
                      self.embed1, 
                      self.embed2, 
                      self.embed3, 
                      self.embed4, 
                      self.embed5, 
                      self.embed6, 
                      self.embed7]

        if embedding_file is not None:
            with open(embedding_file, "rb") as file:
                embedding_dict = pickle.load(file)

        if len(embedding_dict) > 0:
            for i in range(len(self.embed)):
                self.embed[i].load_state_dict(embedding_dict[i])
                for param in self.embed[i].parameters():
                    param.requires_grad = False

        self.main = nn.Sequential(
            nn.Linear(129, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, cate_inputs, num_inputs):
        embeddings = []
        for i in range(len(self.embed)):
            embeddings.append(self.embed[i](cate_inputs[:, i]))
        embedding = torch.cat(embeddings, 1)
        inputs = torch.cat((embedding, num_inputs), 1)
        return self.main(inputs)
    

class EmbeddingModel():
    def __init__(self, 
                 epoch=20, 
                 batch_size=128,
                 learning_rate=0.003,
                 device=torch.device("cuda"),
                 pos_weight=100.0,
                 model=None):
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]).to(device))
        self.model = model.to(device)
        self.widget = [pb.DynamicMessage("epoch"), ' ', 
                  pb.Percentage(), ' ',
                  pb.ETA(), ' ',
                  pb.Timer(), ' ', 
                  pb.DynamicMessage("lossT"), ' ',
                  pb.DynamicMessage("lossV")]

        
    def fit(self,
            source_train,
            source_train_label,
            source_valid,
            source_valid_label, 
            verbose=True):
        
        train_tensor = data.TensorDataset(
            torch.from_numpy(source_train[:, :8]).long(),
            torch.from_numpy(source_train[:, 8:]).float(),
            torch.from_numpy(source_train_label.reshape(-1,1)).float()
        )
        
        valid_tensor = data.TensorDataset(
            torch.from_numpy(source_valid[:, :8]).long(),
            torch.from_numpy(source_valid[:, 8:]).float(),
            torch.from_numpy(source_valid_label.reshape(-1,1)).float()
        )
        
        train_loader = data.DataLoader(train_tensor, batch_size=self.batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_tensor, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        
        best_score = float('inf')
        for num_epoch in range(self.epoch):
            train_per_epoch = source_train.shape[0] / self.batch_size
            iter_per_epoch = (source_train.shape[0] + source_valid.shape[0]) // self.batch_size

            if verbose:
                timer = pb.ProgressBar(widgets=self.widget, max_value=iter_per_epoch+1).start()
            if num_epoch==0 and not verbose:
                timer = pb.ProgressBar(widgets=self.widget, max_value=iter_per_epoch*self.epoch+1).start()

            # train step
            self.model.train(True)
            cum_loss = 0
            for num_batch, (cate_source_dataset, source_dataset, source_label) in enumerate(train_loader):
                cate_source_dataset = cate_source_dataset.to(self.device)
                source_dataset = source_dataset.to(self.device)
                source_label = source_label.to(self.device)
                
                percentage = (num_epoch * train_per_epoch + num_batch) / (self.epoch * train_per_epoch)
                optimizer = self.optimizer_regularization(optimizer, 
                                                     [self.learning_rate],
                                                     10, percentage, 0.75)
                
                self.model.zero_grad()
                
                # prediction loss
                prediction = self.model(cate_source_dataset, source_dataset)
                l = self.loss(prediction, source_label)
                cum_loss += l.cpu().detach().data
                l.backward(retain_graph=True)
                optimizer.step()
                if verbose:
                    timer.update(num_batch+1, epoch=num_epoch+1, lossT=cum_loss.item()/(num_batch+1))
                else:
                    timer.update(num_epoch*iter_per_epoch+num_batch+1, epoch=num_epoch+1, lossT=cum_loss.item()/(num_batch+1))
            
            # valid step
            self.model.train(False)
            train_size = num_batch
            cum_loss = 0
            for num_batch, (cate_source_dataset, source_dataset, source_label)  in enumerate(valid_loader):
                cate_source_dataset = cate_source_dataset.to(self.device)
                source_dataset = source_dataset.to(self.device)
                source_label = source_label.to(self.device)

                prediction = self.model(cate_source_dataset, source_dataset)
                l = self.loss(prediction, source_label)
                cum_loss += l.cpu().data

                if verbose:
                    timer.update(train_size+num_batch+1, lossV=cum_loss.item()/(num_batch+1))
                else:
                    timer.update(iter_per_epoch*num_epoch+train_size+num_batch+1, lossV=cum_loss.item()/(num_batch+1))

            if (num_epoch == self.epoch - 1) or verbose:
                timer.finish()

            if cum_loss/(num_batch+1) < best_score:
                best_score = cum_loss/(num_batch+1)
                best_state = self.model.state_dict()

        print("Best valid score: {:.4f}".format(best_score), flush=True)
        self.model.load_state_dict(best_state)
        self.model.eval()
    
    
    def predict(self, target_train):
        data_tensor = data.TensorDataset(
            torch.from_numpy(target_train[:, :8]).long(),
            torch.from_numpy(target_train[:, 8:]).float()
        )
        dataloader = data.DataLoader(data_tensor, batch_size=self.batch_size, shuffle=False)
        
        res = []
        for num_batch, (cate_dataset, dataset,) in enumerate(dataloader):
            cate_dataset = cate_dataset.to(self.device)
            dataset = dataset.to(self.device)
            
            prediction = self.model(cate_dataset, dataset)
            res.extend(prediction.cpu().data)
        probs = 1 / (1 + np.exp(-np.array(res)))
        return probs


    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
    
    def get_embedding_dict(self):
        return [self.model.embed[i].state_dict() for i in range(8)]
    
    def save_embedding_dict(self, path):
        embedding_dict = [self.model.embed[i].state_dict() for i in range(8)]
        with open(path, "wb") as file:
            pickle.dump(embedding_dict, file)
    
    def optimizer_regularization(self, optimizer, init_lr, alpha, percentage, beta):
        for i in range(len(init_lr)):
            optimizer.param_groups[i]["lr"] = init_lr[i] * (1 + alpha * percentage)**(-beta)

        return optimizer
