import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.optim as optim
from DNN import DNN
from Data import CSVData
from sklearn.model_selection import train_test_split

class train_model:
    def Loss(y, y_pred, delta=0.5): #MSE loss
        a = torch.mean(torch.abs(y - y_pred))
        if a <= delta:
            return 0.5*a**2
        else:
            return delta*(a-0.5*delta)

    def train(train_data, net, optimizer, test_data = None, epochs = 300):
        losses =[]
        test_losses = []
        
        if test_data != None:
            inputs = []
            labels = []
            for i in test_data.file_names:
                X_test, Y_test = test_data.load_data(i)
                inputs.append(X_test)
                labels.append(Y_test)
            test_inputs = torch.Tensor(np.array(inputs))
            test_labels = torch.Tensor(np.log(labels))
            del inputs, labels, X_test, Y_test

        for epoch in range(epochs):

            epoch_loss = []   
            running_loss = 0.0

            for i, batch in enumerate(train_data.generate_data()):

                X, Y = batch
                inputs = torch.Tensor(np.array(X))
                labels = torch.Tensor(np.log(np.array(Y)))
                del X, Y
                
                outputs =net(inputs)
                optimizer.zero_grad()
                loss = train_model.Loss(labels, outputs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_loss.append(loss.item())
                if i % 2000 == 1999:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
            print('Epoch ' + str(epoch+1) +': ' + str(loss.item()))
            losses.append(loss.item())
            if test_data != None:
                test_losses.append((train_model.Loss(test_labels, net(test_inputs)).item()))

        print('Finished Training')
        return losses, test_losses