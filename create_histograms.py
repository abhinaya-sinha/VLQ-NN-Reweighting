import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from Data import CSVData

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model_path = '/projects/bbhj/asinha15/VLQ-NN-Reweighting/main/trained_models/model_scripted7.pt'
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

X, Y = train_data.load_data_many()
inputs = torch.Tensor(np.array(X)).to(device)
labels = torch.Tensor(np.log(np.array(Y))).to(device)
del X, Y

with torch.no_grad():
    out=torch.reshape(model(inputs), [9000000]).to('cpu')

df = pd.DataFrame(inputs.to('cpu'))
df.columns = features

Mvlq = np.sqrt((df['e_out2'] + df['e_out3'])**2 - (df['px_out2'] + df['px_out3'])**2 - (df['py_out2']+df['py_out3'])**2 - (df['pz_out2']+df['pz_out3'])**2)

toGraph = pd.DataFrame({ 'Mvlq' : Mvlq.values, 
           'actual f_rwt' : np.exp(labels.cpu().numpy()),
           'predicted f_rwt' : np.exp(out.numpy())
})

x = ['Msim', 'Gsim', 'Mtarget', 'Gtarget']
for i in x:
    toGraph[i] = df[i].values

del df

Msimuniq=toGraph['Msim'].unique()
Mtargetuniq=toGraph['Mtarget'].unique()
Gsimuniq = toGraph['Gsim'].unique()
Gtargetuniq=toGraph['Gtarget'].unique()
for Msim in Msimuniq:
    Msimidx = set(toGraph.index[toGraph['Msim']==Msim])
    for Gsim in Gsimuniq:
        Gsimidx = set(toGraph.index[toGraph['Gsim']==Gsim])
        for Mtarget in Mtargetuniq:
            Mtargetidx = set(toGraph.index[toGraph['Mtarget']==Mtarget])
            for Gtarget in Gtargetuniq:
                Gtargetidx = set(toGraph.index[toGraph['Gtarget']==Gtarget])
                idx = list(Msimidx & Gsimidx & Mtargetidx & Msimidx)
                plt.hist([toGraph.at[i,'Mvlq'] for i in idx], bins = 60, weights = [toGraph.at[i,'actual f_rwt'] for i in idx], label = 'MadGraph', alpha=1)
                plt.hist([toGraph.at[i,'Mvlq'] for i in idx], bins = 60, weights = [toGraph.at[i, 'predicted f_rwt'] for i in idx], label = 'DNN', alpha=0.75)
                plt.xlabel('Mvlq')
                plt.legend()
                title = 'M' + str(int(Msim/100)) + 'G' + str(int(Gsim*100/Msim) if Gsim*100/Msim >=100 else ('0' + str(int(Gsim*100/Msim)) if Gsim*100/Msim >= 10 else '00' + str(int(Gsim*100/Msim)))) + '(s)' + ' M' + (str(int(Mtarget/100)) if Mtarget/100 >= 10 else '0'+str(int(Mtarget/100))) + 'G' + str(int(Gtarget*100/Mtarget) if Gtarget*100/Mtarget >= 100 else ('0' + str(int(Gtarget*100/Mtarget)) if Gtarget*100/Mtarget >= 10 else '00' + str(int(Gtarget*100/Mtarget)))) + '(r)'
                plt.title(title)
                plt.savefig('/projects/bbhj/asinha15/VLQ-NN-Reweighting/main/histograms/' + title.replace(' ', '') + '.png')
                plt.close()
                del idx
            del Mtargetidx
        del Gsimidx
    del Msimidx
