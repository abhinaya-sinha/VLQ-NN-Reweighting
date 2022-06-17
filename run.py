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

features = ['mode',
'Msim',
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
VLQData = CSVData(batch_size=2048, features_name=features, labels_name=label, features_to_rescale= features_to_rescale, file_names=['/projects/bbhj/asinha15/train_'+str(i)+'.csv' for i in range(0,10)])
test_data = CSVData(batch_size=1024, features_name=features, labels_name=label, features_to_rescale= features_to_rescale, file_names=['/projects/bbhj/asinha15/test_' + str(i) + '.csv' for i in range(0,5)])

net = DNN(Layers=[23,32, 64, 16, 8]).build_model()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
epochs=150

losses, test_losses, accuracies = train_model.train(train_data=VLQData, test_data = test_data, net = net, optimizer=optimizer, epochs=epochs)

model_scripted = torch.jit.script(net)
model_scripted.save('trained_models/model_scripted4.pt')

plt.plot(np.linspace(0,epochs, epochs), losses, label = 'train loss')
plt.yscale('log')
plt.plot(np.linspace(0, epochs, epochs), test_losses, label = 'test loss')
plt.yscale('log')
plt.legend()
plt.savefig('plot4.png')
plt.show()
plt.close()

plt.plot(np.linspace(0,epochs,epochs), accuracies)
plt.title('test accuracy')
plt.savefig('accuracy1.png')
plt.show()
plt.close()