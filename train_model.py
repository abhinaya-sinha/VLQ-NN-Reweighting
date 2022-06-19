import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.optim as optim
from DNN import DNN
from Data import CSVData

class train_model:
    def Loss(y, y_pred, delta=0.5): #MSE loss
        loss_function = nn.HuberLoss(delta=0.5)
        return loss_function(y_pred.view(y.size()), y)

    def train(train_data, net, optimizer, test_data = None, epochs = 300, device='cuda'):
        losses =[]
        test_losses = []
        
        if test_data != None:
            X_test, Y_test = test_data.load_data_many()
            test_inputs = torch.Tensor(np.array(X_test)).to(device)
            test_labels = torch.Tensor(np.array(np.log(Y_test))).to(device)
            del X_test, Y_test
            accuracies = []

        for epoch in range(epochs):

            epoch_loss = []
            running_loss = 0.0
            for X, Y in train_data.generate_data():

                inputs = torch.Tensor(np.array(X)).to(device)
                labels = torch.Tensor(np.log(np.array(Y))).to(device)
                del X, Y
                
                outputs =net(inputs)
                optimizer.zero_grad()
                loss = train_model.Loss(labels, outputs)
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
                    test_out=torch.reshape(net(test_inputs), (test_labels.size(dim=0),)).to(device)
                    test_losses.append((train_model.Loss(test_labels, test_out).item()))
                    accuracies.append(1-torch.mean(torch.abs((test_labels-test_out)/test_labels)).item())
                    del test_out

        print('Finished Training')
        return losses, test_losses, accuracies