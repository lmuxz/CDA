import torch
import numpy as np
import progressbar as pb

from torch import nn
from torch.utils import data


class Adversarial(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, inputs):
        return self.main(inputs)

class DANN():
    def __init__(self, model, device=torch.device("cuda")):
        self.device = device
        self.model = model.to(device)
    

    def fit(self, xs, ys, xt, xv, yv, epoch=20, batch_size=16, lr=0.001, beta=0.1, alpha:int=1, min_mmd=0, early_stop=True, verbose=True):
        adv = Adversarial(self.model.input_layer[-2].out_features).to(self.device)

        train_tensor = data.TensorDataset(
            torch.from_numpy(xs).float(),
            torch.from_numpy(ys).reshape(-1,1).float(),
            torch.from_numpy(xt).float()
        )

        valid_tensor = data.TensorDataset(
            torch.from_numpy(xv).float(),
            torch.from_numpy(yv).reshape(-1,1).float()
        )

        train_loader = data.DataLoader(train_tensor, batch_size=batch_size)
        valid_loader = data.DataLoader(valid_tensor, batch_size=batch_size)

        optim = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_adv = torch.optim.Adam(adv.parameters(), lr=lr, betas=(0.9, 0.999))

        widget = [pb.DynamicMessage("epoch"), ' ', 
                    pb.Percentage(), ' ',
                    pb.ETA(), ' ',
                    pb.Timer(), ' ', 
                    pb.DynamicMessage("lossT"), ' ',
                    pb.DynamicMessage("lossV"), ' ',
                    pb.DynamicMessage("mmd")]

        best_risk = float('inf')
        for e in range(epoch):
            iteration_per_epoch = int(xs.shape[0] / batch_size) + 1

            if verbose:
                timer = pb.ProgressBar(widgets=widget, maxval=iteration_per_epoch).start()

            self.model.train(True)
            adv.train(True)
            cum_loss = 0
            cum_mmd = 0
            for b, (source_dataset, source_label, target_dataset) in enumerate(train_loader):
                source_dataset = source_dataset.to(self.device)
                source_label = source_label.to(self.device)
                target_dataset = target_dataset.to(self.device)

                percentage = (e * iteration_per_epoch + b) / (epoch * iteration_per_epoch)
                optim = self.optimizer_regularization(optim, [lr], 10, percentage, 0.75)
                optim_adv = self.optimizer_regularization(optim_adv, [lr], 10, percentage, 0.75)

                output = self.model(source_dataset)
                self.model.zero_grad()
                l = nn.BCEWithLogitsLoss()(output, source_label)
                l.backward(retain_graph=True)
                optim.step()
                cum_loss += l.cpu().detach().data

                if beta==0:
                    mmd = 0
                else:
                    ones = torch.ones(source_dataset.size()[0], device=self.device)
                    zeros = torch.zeros(target_dataset.size()[0], device=self.device)
                    label = torch.cat((ones, zeros)).reshape(-1, 1).float()
                    fake_label = torch.cat((zeros, ones)).reshape(-1, 1).float()

                    source_hidden_rep = self.model.hidden_rep
                    _ = self.model(target_dataset)
                    target_hidden_rep = self.model.hidden_rep

                    output_adv = adv(torch.cat((source_hidden_rep, target_hidden_rep)))

                    self.model.zero_grad()
                    l_adv = beta * nn.BCEWithLogitsLoss()(output_adv, fake_label)
                    l_adv.backward(retain_graph=True)
                    optim.step()

                    if b % alpha == 0:
                        adv.zero_grad()
                        l_adv = alpha * nn.BCEWithLogitsLoss()(output_adv, label)
                        l_adv.backward()
                        optim_adv.step()
                    cum_mmd += l_adv.cpu().detach().data
                avg_mmd = float(cum_mmd)/(b+1)
                if verbose:
                    timer.update(b+1, epoch=e+1, lossT=float(cum_loss)/(b+1), mmd=avg_mmd)
            
            self.model.train(False)
            adv.train(False)
            cum_loss = 0
            for b, (source_dataset, source_label) in enumerate(valid_loader):
                source_dataset = source_dataset.to(self.device)
                source_label = source_label.to(self.device)

                output = self.model(source_dataset)
                l = nn.BCEWithLogitsLoss()(output, source_label)
                cum_loss += l.cpu().detach().data

                if verbose:
                    timer.update(iteration_per_epoch, epoch=e+1, lossV=float(cum_loss)/(b+1))
            if verbose:
                timer.finish()

            if (cum_loss/(b+1) < best_risk) and (avg_mmd > min_mmd):
                best_risk = cum_loss/(b+1)
                best_state = self.model.state_dict()
            else:
                if early_stop:
                    if verbose:
                        print("Early Stop", flush=True)
                    break
        if verbose:
            print("Best valid risk: {:.4f}".format(best_risk), flush=True)
        self.model.load_state_dict(best_state)
        self.model.eval()


    def predict(self, target_train, batch_size=128):
        probs = self.predict_prob(target_train, batch_size)
        return probs > 0.5


    def predict_prob(self, target_train, batch_size=128):
        data_tensor = data.TensorDataset(
            torch.from_numpy(target_train).float()
        )
        dataloader = data.DataLoader(data_tensor, batch_size=batch_size, shuffle=False)
        
        res = []
        for num_batch, (dataset,) in enumerate(dataloader):
            dataset = dataset.to(self.device)
            prediction = self.model(dataset)
            res.extend(prediction.cpu().detach().data)
        probs = 1 / (1 + np.exp(-np.array(res)))
        return probs

    def optimizer_regularization(self, optimizer, init_lr, alpha, percentage, beta):
        for i in range(len(init_lr)):
            optimizer.param_groups[i]["lr"] = init_lr[i] * (1 + alpha * percentage)**(-beta)

        return optimizer