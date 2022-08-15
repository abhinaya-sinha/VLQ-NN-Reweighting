import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.optim as optim
from distribution_DNN import distribution_DNN
from Data import CSVData

class train_model:

    def loss_fn(net, dist, y):
        dist = dist
        x = y
        if net.prob_dist == 'gaussian':
            L = torch.mean(-(-0.5*torch.log(2*np.pi)-0.5*torch.log(dist[:,1])-((x-dist[:,0])**2)/(2*dist[1]**2)))
        elif net.prob_dist == 'poisson':
            L = torch.mean(-(-dist-torch.log(np.math.factorial(x)+torch.log(dist)*x)))
        elif net.prob_dist == 'weibull':
            L = torch.mean(-(-torch.log(dist[:,0])-dist[:,0]*torch.log(dist[:,1])-(x/dist[:,1])**dist[:,0]+(dist[:,0]-1)*torch.log(x)))
        elif net.prob_dist == 'continuous bernoulli':
            n = torch.log(dist/(1-dist))
            L = torch.mean(-(n*x - n*torch.log(torch.exp(n)-1)+n*torch.log(n)), keepdim=True)
        else:
            raise Exception('that probability distribution is not supported. the options are / '+(p + ' / ' for p in net.possible_prob_dists))
        return L

    def train(train_data, net, optimizer, test_data = None, val_data = None, epochs = 300, device='cuda'):
        losses =[]
        test_losses = []
        val_losses = []
        model = net.Model
        
        if test_data != None:
            X_test, Y_test = test_data.load_data_many()
            test_inputs = torch.Tensor(np.array(X_test)).to(device)
            test_labels = torch.Tensor(np.array(np.log(Y_test))).to(device)
            del X_test, Y_test
        
        if val_data != None:
            X_val, Y_val = val_data.load_data_many()
            val_inputs = torch.Tensor(np.array(X_val)).to(device)
            val_labels = torch.Tensor(np.array(np.log(Y_val))).to(device)
            del X_val, Y_val

        for epoch in range(epochs):

            epoch_loss = []
            running_loss = 0.0
            for X, Y in train_data.generate_data():

                inputs = torch.Tensor(np.array(X)).to(device)
                labels = torch.Tensor(np.log(np.array(Y))).to(device)
                del X, Y
                
                outputs =model(inputs)
                optimizer.zero_grad()
                loss = train_model.loss_fn(net, outputs, labels)
                loss.backward()
                optimizer.step()
                del outputs, inputs, labels
                epoch_loss.append(loss.item())
                del loss
                    
            loss = np.mean(epoch_loss)
            print('Epoch ' + str(epoch+1) +': ' + str(loss))
            losses.append(loss)
            del loss, epoch_loss
            
            if test_data != None:
                with torch.no_grad():
                    test_out=model(test_inputs).to(device)
                    test_losses.append((train_model.loss_fn(net, test_out, test_labels)).item())
                    del test_out
            
            if val_data != None:
                with torch.no_grad():
                    val_out=model(val_inputs).to(device)
                    val_losses.append(train_model.loss_fn(net, val_out, val_labels).item())
                    del val_out
            
        print('Finished Training')
        return losses, test_losses, val_losses