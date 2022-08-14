import torch
import torch.nn as nn

class distribution_DNN(nn.Module):
    def __init__(self, Layers=[25, 16], activation = nn.LeakyReLU(), last_activation = nn.Softplus(), X_mean = 0, X_std = 1, device = 'cuda', prob_dist = 'gaussian'):
        super().__init__()
        self.Layers = Layers
        self.activation = activation
        self.device = device
        self.activation = activation
        self.last_activation = last_activation
        self.device = device
        self.Model = self.build_model().to(device)
        self.X_mean = X_mean 
        self.X_std = X_std
        self.possible_prob_dists = [ 'gaussian',
            'poisson',
            'weibull',
            'continuous bernoulli']
        self.prob_dist = prob_dist
        if prob_dist not in self.possible_prob_dists:
            raise Exception('that probability distribution is not supported. the options are / '+(p + ' / ' for p in self.possible_prob_dists))

        def build_model(self):
            net = nn.Sequential()
            for ii in range(len(self.Layers)-2):
                this_module = nn.Linear(self.Layers[ii], self.Layers[ii+1])
                nn.init.xavier_normal_(this_module.weight)
                net.add_module("Linear" + str(ii), this_module)
                net.add_module("Activation" + str(ii), self.activation)
            self.add_last_module(net)
            return net 

        def add_last_module(self, net):
            if self.prob_dist == 'gaussian':
                n_out = 2 #mean, standard deviation
            elif self.prob_dist == 'poisson':
                n_out = 1 #lambda
            elif self.prob_dist == 'weibull':
                n_out = 2 #k, lambda
            elif self.prob_dist == 'continuous bernoulli':
                n_out = 1 #lambda
            else:
                raise Exception('that probability distribution is not supported. the options are / '+(p + ' / ' for p in self.possible_prob_dists))
            this_module = nn.Linear(self.Layers[-2], n_out, True)
            nn.init.xavier_normal_(this_module.weight)
            net.add_module("Linear_last", this_module)
            last_activation = self.last_activation
            if last_activation != None:
                net.add_module("Activation_last", last_activation)
