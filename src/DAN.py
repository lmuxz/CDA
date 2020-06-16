import torch
import numpy as np
import progressbar as pb

from torch import nn
from torch.utils import data


class DanseNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, inputs):
        self.hidden_rep = self.input_layer(inputs)
        return self.output_layer(self.hidden_rep)


class DAN():
    def __init__(self, model, device=torch.device("cuda")):
        self.device = device
        self.model = model.to(device)
    

    def fit(self, xs, ys, xt, xv, yv, epoch=20, batch_size=16, lr=0.001, beta=0.1, early_stop=True, verbose=True):
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
            cum_loss = 0
            cum_mmd = 0
            for b, (source_dataset, source_label, target_dataset) in enumerate(train_loader):
                source_dataset = source_dataset.to(self.device)
                source_label = source_label.to(self.device)
                target_dataset = target_dataset.to(self.device)
                
                self.model.zero_grad()

                output = self.model(source_dataset)
                l = nn.BCEWithLogitsLoss()(output, source_label)
                l.backward(retain_graph=True)
                cum_loss += l.cpu().detach().data

                if beta==0:
                    mmd = 0
                else:
                    source_hidden_rep = self.model.hidden_rep
                    _ = self.model(target_dataset)
                    target_hidden_rep = self.model.hidden_rep

                    mmd = beta * self.mmd_rbf_noaccelerate(source_hidden_rep, target_hidden_rep, 
                                                            kernel_mul=2.0, kernel_num=4, fix_sigma=None)
                    mmd.backward()
                    cum_mmd += mmd.cpu().detach().data

                percentage = (e * iteration_per_epoch + b) / (epoch * iteration_per_epoch)
                optim = self.optimizer_regularization(optim, [lr], 10, percentage, 0.75)
                optim.step()
                if verbose:
                    timer.update(b+1, epoch=e+1, lossT=float(cum_loss)/(b+1), mmd=float(cum_mmd)/(b+1))
            
            self.model.train(False)
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

            if cum_loss/(b+1) < best_risk:
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


    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)#/len(kernel_val)


    def mmd_rbf_noaccelerate(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

    def optimizer_regularization(self, optimizer, init_lr, alpha, percentage, beta):
        for i in range(len(init_lr)):
            optimizer.param_groups[i]["lr"] = init_lr[i] * (1 + alpha * percentage)**(-beta)

        return optimizer