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
        a = torch.mean(torch.abs(y - y_pred))
        if a <= delta:
            return 0.5*a**2
        else:
            return delta*(a-0.5*delta)

    def train(train_data, net, optimizer, test_data = None, epochs = 300):
        losses =[]
        test_losses = []
        
        if test_data != None:
            X_test, Y_test = test_data.load_data_many()
            test_inputs = torch.Tensor(np.array(X_test))
            test_labels = torch.Tensor(np.array(Y_test))
            del X_test, Y_test

        for epoch in range(epochs):

            epoch_loss = []
            total_test_losses=[]
            running_loss = 0.0

            for X, Y in train_data.generate_data():

                inputs = torch.Tensor(np.array(X))
                labels = torch.Tensor(np.log(np.array(Y)))
                del X, Y
                
                outputs =net(inputs)
                optimizer.zero_grad()
                loss = train_model.Loss(labels, outputs)
                loss.backward()
                optimizer.step()
                del outputs, inputs, labels

                running_loss += loss.item()
                epoch_loss.append(loss.item())
                del loss
                
                if test_data != None:
                    with torch.no_grad():
                        indexes = [np.random.randint(0, test_labels.size(0)) for i in range(0, 1024)]
                        test_input_subsection = test_inputs[indexes]
                        test_out=net(test_input_subsection)
                        total_test_losses.append((train_model.Loss(test_labels[indexes], test_out).item()))
                        del test_out, test_input_subsection, indexes
                    
            loss = np.mean(epoch_loss)
            print('Epoch ' + str(epoch+1) +': ' + str(loss))
            losses.append(loss)
            del loss, epoch_loss
            
            if test_data != None:
                test_losses.append(np.mean(total_test_losses))
                del total_test_losses
                
        print('Finished Training')
        return losses, test_losses