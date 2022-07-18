import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import Data
from Data import CSVData
from train_model import train_model
from DNN import DNN

if torch.cuda.is_available():
	device = torch.device("cuda") 
else:
	device = torch.device("cpu")
	raise Exception("GPU not found")


features = ['mode',
'Msim',
'Gsim',
'weight',
'pz_in1',
'pid_in1', 
'pz_in2',
'pid_in2',
'px_out1',
'py_out1',
'pz_out1',
'e_out1',
'pid_out1',
'px_out2',
'py_out2',
'pz_out2',
'e_out2',
'pid_out2',
'px_out3',
'py_out3',
'pz_out3',
'e_out3',
'pid_out3',
'px_out4',
'py_out4',
'pz_out4',
'e_out4',
'pid_out4',
'Mtarget',
'Gtarget',]
label = 'f_rwt'
features_to_rescale = ['Msim',
'Gsim',
'pz_in1', 
'pz_in2',
'px_out1',
'py_out1',
'pz_out1',
'e_out1',
'px_out2',
'py_out2',
'pz_out2',
'e_out2',
'px_out3',
'py_out3',
'pz_out3',
'e_out3',
'px_out4',
'py_out4',
'pz_out4',
'e_out4',
'Mtarget',
'Gtarget']
VLQData = CSVData(batch_size=2048, features_name=features, labels_name=label, features_to_rescale= features_to_rescale, file_names=['/projects/bbhj/asinha15/train_'+str(i)+'.csv' for i in range(0,8)])
test_data = CSVData(batch_size=1024, features_name=features, labels_name=label, features_to_rescale= features_to_rescale, file_names=['/projects/bbhj/asinha15/test_' + str(i) + '.csv' for i in range(0,10) if i%2==0])
val_data = CSVData(batch_size=2048, features_name=features, labels_name=label, features_to_rescale= features_to_rescale, file_names=['/projects/bbhj/asinha15/train_'+str(i)+'.csv' for i in range(8,10)])

net = DNN(Layers=[30, 32, 64, 16, 8, 4], device=device).Model
optimizer = optim.Adam(net.parameters(), lr=1e-3)
epochs=200
loss_functions = [nn.MSELoss, nn.HuberLoss(delta=0.5)]
loss_functions_names = ['MSE Loss', 'Huber Loss']

for loss, loss_function_name in loss_functions, loss_functions_names:
    losses, test_losses, val_losses, accuracies = train_model.train(train_data=VLQData, test_data = test_data, val_data = val_data, net = net, optimizer=optimizer, epochs=epochs, device=device)

    model_scripted = torch.jit.script(net)
    model_scripted.save('/projects/bbhj/asinha15/VLQ-NN-Reweighting/main/trained_models/'+ loss_functions_names+'1.pt')

    plt.plot(np.linspace(0,epochs, epochs), losses, label = 'train loss')
    plt.yscale('log')
    plt.plot(np.linspace(0, epochs, epochs), test_losses, label = 'test loss')
    plt.yscale('log')
    plt.plot(np.linspace(0, epochs, epochs), val_losses, label = 'validation loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('plots/LossFunctionPlots/'+loss_function_name + '.png')
    plt.show()
    plt.close()
    plt.plot(np.linspace(0,epochs,epochs), accuracies)
    plt.title('1-relative absolute error')
    plt.savefig('plots/RAE/'+loss_function_name + '.png')
    plt.show()
    plt.close()