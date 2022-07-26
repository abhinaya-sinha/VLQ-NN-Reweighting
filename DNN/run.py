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


features = ['Msim',
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
'Gtarget',
'mode-W', 
'mode-H', 
'mode-Z',]
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
'Gtarget',]
VLQData = CSVData(batch_size=2048, features_name=features, labels_name=label, features_to_rescale= features_to_rescale, file_names=['/projects/bbhj/asinha15/train_'+str(i)+'.csv' for i in range(0,8)])
test_data = CSVData(batch_size=1024, features_name=features, labels_name=label, features_to_rescale= features_to_rescale, file_names=['/projects/bbhj/asinha15/test_' + str(i) + '.csv' for i in range(0,10)])
val_data = CSVData(batch_size=2048, features_name=features, labels_name=label, features_to_rescale= features_to_rescale, file_names=['/projects/bbhj/asinha15/train_'+str(i)+'.csv' for i in range(8,10)])

net = DNN(Layers=[25, 32, 64, 32, 32, 16, 8, 4], device=device).Model
optimizer = optim.Adam(net.parameters(), lr=1e-3)
epochs=300

losses, test_losses, val_losses, accuracies = train_model.train(train_data=VLQData, test_data = test_data, val_data = val_data, net = net, optimizer=optimizer, epochs=epochs, device=device)

model_scripted = torch.jit.script(net)
model_scripted.save('/projects/bbhj/asinha15/VLQ-NN-Reweighting/main/trained_models/[25, 32, 64, 32, 32, 16, 8, 4].pt')

plt.plot(np.linspace(0,len(losses), len(losses)), losses, label = 'train loss')
plt.yscale('log')
plt.plot(np.linspace(0, len(losses), len(losses)), test_losses, label = 'test loss')
plt.yscale('log')
plt.plot(np.linspace(0, len(losses), len(losses)), val_losses, label = 'validation loss')
plt.yscale('log')
plt.legend()
plt.savefig('/projects/bbhj/asinha15/VLQ-NN-Reweighting/main/plots/LossFunctionPlots/[25, 32, 64, 32, 32, 16, 8, 4].png')
plt.show()
plt.close()

plt.plot(np.linspace(0,len(losses),len(losses)), accuracies)
plt.title('validation accuracy')
plt.savefig('/projects/bbhj/asinha15/VLQ-NN-Reweighting/main/plots/RAE/[25, 32, 64, 64, 32, 16, 8, 4].png')
plt.show()
plt.close()
