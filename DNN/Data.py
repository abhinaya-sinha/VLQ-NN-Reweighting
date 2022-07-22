import numpy as np
import pandas as pd
import os
import time
from threading import Thread
import itertools
import torch

class Data(object):
    """Class providing an interface to the input training data.
        Derived classes should implement the load_data function.
        Attributes:
          file_names: list of data files to use for training
          batch_size: size of training batches
    """

    def __init__(self, batch_size, cache=None, spectators=False):
        """Stores the batch size and the names of the data files to be read.
            Params:
              batch_size: batch size for training
        """
        self.batch_size = batch_size
        self.caching_directory = cache if cache else os.environ.get('GANINMEM','')
        self.spectators = spectators
                
    def generate_data(self):
       """Yields batches of training data until none are left."""
       leftovers = None
       for cur_file_name in self.file_names:
           if self.spectators:
               cur_file_features, cur_file_labels, cur_file_spectators = self.load_data(cur_file_name)
           else:
               cur_file_features, cur_file_labels = self.load_data(cur_file_name)
           # concatenate any leftover data from the previous file
           if leftovers is not None:
               cur_file_features = self.concat_data( leftovers[0], cur_file_features )
               cur_file_labels = self.concat_data( leftovers[1], cur_file_labels )
               if self.spectators:
                   cur_file_spectators = self.concat_data( leftovers[2], cur_file_spectators)                   
               leftovers = None
           num_in_file = self.get_num_samples( cur_file_features )

           for cur_pos in range(0, num_in_file, self.batch_size):
               next_pos = cur_pos + self.batch_size 
               if next_pos <= num_in_file:
                   if self.spectators:
                       yield ( self.get_batch( cur_file_features, cur_pos, next_pos ),
                               self.get_batch( cur_file_labels, cur_pos, next_pos ),
                               self.get_batch( cur_file_spectators, cur_pos, next_pos ) )
                   else:
                       yield ( self.get_batch( cur_file_features, cur_pos, next_pos ),
                               self.get_batch( cur_file_labels, cur_pos, next_pos ) )
               else:
                   if self.spectators:
                       leftovers = ( self.get_batch( cur_file_features, cur_pos, num_in_file ),
                                     self.get_batch( cur_file_labels, cur_pos, num_in_file ),
                                     self.get_batch( cur_file_spectators, cur_pos, num_in_file) )
                   else:
                       leftovers = ( self.get_batch( cur_file_features, cur_pos, num_in_file ),
                                     self.get_batch( cur_file_labels, cur_pos, num_in_file ) )

    def count_data(self):
        """Counts the number of data points across all files"""
        num_data = 0
        for cur_file_name in self.file_names:
            cur_file_features, cur_file_labels = self.load_data(cur_file_name)
            num_data += self.get_num_samples( cur_file_features )
        return num_data

    def is_numpy_array(self, data):
        return isinstance( data, np.ndarray )

    def get_batch(self, data, start_pos, end_pos):
        """Input: a numpy array or list of numpy arrays.
            Gets elements between start_pos and end_pos in each array"""
        if self.is_numpy_array(data):
            return data[start_pos:end_pos] 
        else:
            return [ arr[start_pos:end_pos] for arr in data ]

    def concat_data(self, data1, data2):
        """Input: data1 as numpy array or list of numpy arrays.  data2 in the same format.
           Returns: numpy array or list of arrays, in which each array in data1 has been
             concatenated with the corresponding array in data2"""
        if self.is_numpy_array(data1):
            return np.concatenate( (data1, data2) )
        else:
            return [ self.concat_data( d1, d2 ) for d1,d2 in zip(data1,data2) ]

    def get_num_samples(self, data):
        """Input: dataset consisting of a numpy array or list of numpy arrays.
            Output: number of samples in the dataset"""
        if self.is_numpy_array(data):
            return len(data)
        else:
            return len(data)

    def load_data(self, in_file):
        """Input: name of file from which the data should be loaded
            Returns: tuple (X,Y) where X and Y are numpy arrays containing features 
                and labels, respectively, for all data in the file
            Not implemented in base class; derived classes should implement this function"""
        raise NotImplementedError

class CSVData(Data):

    def __init__(self, batch_size, cache=None, preloading=False, features_name='features', features_to_rescale= [], labels_name='labels', spectators_name = None, file_names=[]):
        """Initializes and stores names of feature and label datasets"""
        super(CSVData, self).__init__(batch_size,cache,(spectators_name is not None))
        self.features_name = features_name
        self.labels_name = labels_name        
        self.spectators_name = spectators_name
        self.file_names = file_names 
        self.features_to_rescale=features_to_rescale
    def load_data(self, in_file_name):
        """Loads numpy arrays (or list of numpy arrays) from csv file.
        """
        csv_file = pd.read_csv(in_file_name)
        csv_file[self.features_to_rescale].divide(1000)
        Y = csv_file[self.labels_name].to_numpy()/(csv_file['weight'].to_numpy())
        self.features_name.remove('weight')
        X = csv_file[self.features_name].to_numpy()
        if self.spectators_name is not None:
            Z = csv_file[self.spectators_name].to_numpy()
        if self.spectators_name is not None:
            return X,Y, Z
        else:
            return X,Y
    def load_data_many(self):
        """Loads numpy arrays (or list of numpy arrays) from several csv files.
        """
        csv_file = pd.concat(pd.read_csv(file) for file in self.file_names)
        csv_file[self.features_to_rescale].divide(1000)
        Y = csv_file[self.labels_name].to_numpy()/(csv_file['weight'].to_numpy())
        try:
            self.features_name.remove('weight')
        except:
            pass
        X = csv_file[self.features_name].to_numpy()
        if self.spectators_name is not None:
            Z = csv_file[self.spectators_name].to_numpy()
        if self.spectators_name is not None:
            return X,Y, Z
        else:
            return X,Y