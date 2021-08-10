import math
from os.path import abspath, dirname, join

import numpy as np
import mne
import tensorflow as tf
import torch

mne.set_log_level(verbose=False)

class EDFData():
    def __init__(self, path, channels=None, binary=False):
        self.path = path
        self.channels = channels if channels else 'all'
        self.binary = binary
        self.epochs, self.sampling_rate = self.get_epochs(path)
        self.id_to_class_dict_original = {value-1:key for key, value in self.epochs.event_id.items()}
        self.id_to_class_dict = {0:'Sleeping', 1:'Awake'} if self.binary else self.id_to_class_dict_original
        self.mean, self.std = self.iterative_mean_std()

    def get_epochs(self, path):
        data = mne.io.read_raw_edf(path)
        sampling_rate = data.info['sfreq']
        events, events_id = mne.events_from_annotations(data, regexp='Sleep stage [A-Z]\d*')

        tmax = 30. - 1. / sampling_rate  # tmax is included
        epochs = mne.Epochs(raw=data, 
                            events=events,
                            event_id=events_id,
                            tmin=0., 
                            tmax=tmax, 
                            baseline=None, 
                            event_repeated='merge',
                            picks=self.channels)

        epochs.drop_bad()
        return epochs, sampling_rate

    def iterative_mean_std(self):
        std = 0
        mean = 0
        for i, a in enumerate(self.epochs, 0):
            a = a.mean(axis=-1)
            new_mean = mean + (1/(i+1))*(a-mean)
            std = std + (1/(i+1))*((a-mean)*(a-new_mean)-std)
            mean = new_mean
        return mean, std**(1/2)
    
    @staticmethod
    def refit_std(datasets):
        """
        Takes a list of datasets as input and calculates the common mean and std.
        Then sets their individual mean and std to the common one. 
        This would be the equivalent to standarize them all together.

        Parameters
        ----------
        datasets: list[EDFData]
        """
        common_mean = np.mean([dataset.mean for dataset in datasets], axis=0)
        common_std = np.mean([dataset.std for dataset in datasets], axis=0)

        for dataset in datasets:
            dataset.mean = common_mean
            dataset.std = common_std

class EDFData_TF_old(EDFData, tf.keras.utils.Sequence):
    def __init__(self, path, batch_size, channels=None):
        EDFData.__init__(self, path, channels)
        tf.keras.utils.Sequence.__init__(self)
        self.path = path
        self.batch_size = batch_size
        self.channels = channels if channels else 'all'
        self.epochs, self.sampling_rate = self.get_epochs(path)
        self.id_to_class_dict = {value-1:key for key, value in self.epochs.event_id.items()}

    def __getitem__(self, idx):
        # In TF, should return a full batch

        X = self.epochs[idx * self.batch_size:(idx + 1)*self.batch_size].load_data()._data
        Y = self.epochs[idx * self.batch_size:(idx + 1)*self.batch_size].events[:,-1]-1

        # return tf.squeeze(tf.Tensor(self.epochs[idx].load_data()._data)), tf.Tensor([self.epochs[idx].events[0][-1]])-1
        return X, Y

    def __len__(self):
        # In TF, len should return the number of batches
        return math.ceil(len(self.epochs)/self.batch_size)

class EDFData_TF(EDFData, tf.keras.utils.Sequence):
    def __init__(self, path, batch_size, channels=None):
        EDFData.__init__(self, path, channels)
        tf.keras.utils.Sequence.__init__(self)
        self.batch_size = batch_size

    def __getitem__(self, idx):
        # In TF, __getitem__ should return a full batch

        X = self.epochs[idx * self.batch_size:(idx + 1)*self.batch_size].load_data()._data
        Y = self.epochs[idx * self.batch_size:(idx + 1)*self.batch_size].events[:,-1]-1

        # return tf.squeeze(tf.Tensor(self.epochs[idx].load_data()._data)), tf.Tensor([self.epochs[idx].events[0][-1]])-1
        return (X - self.mean)/self.std, Y

    def __len__(self):
        # In TF, len should return the number of batches
        return math.ceil(len(self.epochs)/self.batch_size)

class EDFData_PTH(EDFData, torch.utils.data.Dataset):
    def __init__(self, path, channels=None, binary=False):
        EDFData.__init__(self, path, channels, binary)
        torch.utils.data.Dataset.__init__(self)
        # self.standarize = std


    def __getitem__(self, idx):
        ## Load data in memory
        X = torch.squeeze(torch.Tensor(self.epochs[idx].get_data()), dim=0)
        ## Standarize data
        X = (X - torch.unsqueeze(torch.tensor(self.mean),-1))/torch.unsqueeze(torch.tensor(self.std),-1)
        ## Get labels
        Y = torch.Tensor([self.epochs[idx].events[0][-1]])-1

        ## If binary, change Sleep stage W for 1 and the rest for 0
        if self.binary:
            cuac = []
            for y in Y:
                if self.id_to_class_dict_original[y.item()]=='Sleep stage W':
                    cuac.append(1)
                else:
                    cuac.append(0)
            Y = torch.tensor(cuac)
        return X, Y
     
    def __len__(self):
        return len(self.epochs)


if __name__ == '__main__':
    rel_path = abspath(join(dirname(__file__)))
    prueba = EDFData_TF(rel_path+"/../Data/PSG1.edf", batch_size=16, channels=['F4'])
    prueba_2 = EDFData_TF_old(rel_path+"/../Data/PSG1.edf", batch_size=16, channels=['F4'])
    prueba_3 = EDFData_PTH(rel_path+"/../Data/PSG1.edf", channels=['F4'])
    print()