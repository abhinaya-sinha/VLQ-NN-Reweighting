import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import Data
from Data import CSVData

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model_path = '/projects/bbhj/asinha15/VLQ-NN-Reweighting/main/trained_models/model_scripted6.pt'
model = torch.jit.load(model_path).to(device)
model.eval()

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

train_data = CSVData(batch_size=1024, features_name=features, labels_name=label, features_to_rescale= features_to_rescale, file_names=['/projects/bbhj/asinha15/train_' + str(i)+'.csv' for i in range(0,10)])
test_data = CSVData(batch_size=1024, features_name=features, labels_name=label, features_to_rescale= features_to_rescale, file_names=['/projects/bbhj/asinha15/test_' + str(i)+'.csv' for i in range(0,10)])

X, Y = train_data.load_data_many()
train_inputs = torch.Tensor(np.array(X)).to(device)
train_labels = torch.Tensor(np.log(np.array(Y))).to(device)
del X, Y

X, Y = test_data.load_data_many()
test_inputs = torch.Tensor(np.array(X)).to(device)
test_labels = torch.Tensor(np.log(np.array(Y))).to(device)
del X, Y

with torch.no_grad():
    train_out=torch.reshape(model(train_inputs), [9000000]).to('cpu')
    test_out=torch.reshape(model(test_inputs), [1500000]).to('cpu')

plt.scatter(train_labels.to('cpu'), train_out, s=0.5, alpha=0.5, label='train data')
plt.scatter(test_labels.to('cpu'), test_out, s=0.5, alpha=0.5, label='test data')
plt.xlabel('labels')
plt.ylabel('predicted')
plt.legend()
plt.savefig('/projects/bbhj/asinha15/VLQ-NN-Reweighting/main/plots/predictedvslabels2.png')
plt.show()