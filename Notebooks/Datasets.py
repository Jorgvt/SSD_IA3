import math
from os.path import abspath, dirname, join

import mne
import tensorflow as tf
import torch

mne.set_log_level(verbose=False)

class EDFData():
    def __init__(self, path, channels=None):
        self.path = path
        self.channels = channels if channels else 'all'
        self.epochs, self.sampling_rate = self.get_epochs(path)
        self.id_to_class_dict = {value-1:key for key, value in self.epochs.event_id.items()}
        self.mean = self.calculate_mean(self.epochs)

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
    
    def calculate_mean(self, epochs):
        mean = 0
        for i,a in enumerate(epochs,0):
            mean = mean + (1/(i+1))*(a.mean(axis=-1)-mean)
        return mean

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
        return X, Y

    def __len__(self):
        # In TF, len should return the number of batches
        return math.ceil(len(self.epochs)/self.batch_size)

class EDFData_PTH(EDFData, torch.utils.data.Dataset):
    def __init__(self, path, channels=None):
        EDFData.__init__(self, path, channels)
        torch.utils.data.Dataset.__init__(self)


    def __getitem__(self, idx):
        return torch.squeeze(torch.Tensor(self.epochs[idx].load_data()._data), dim=1), torch.Tensor([self.epochs[idx].events[0][-1]])-1
    # def __getitem__(self, idx):
    #     return self.epochs[idx]._data, self.epochs[idx].events[0][-1]

    def __len__(self):
        return len(self.epochs)


if __name__ == '__main__':
    rel_path = abspath(join(dirname(__file__)))
    prueba = EDFData_TF(rel_path+"/../Data/PSG1.edf", batch_size=16, channels=['F4'])
    prueba_2 = EDFData_TF_old(rel_path+"/../Data/PSG1.edf", batch_size=16, channels=['F4'])
    prueba_3 = EDFData_PTH(rel_path+"/../Data/PSG1.edf", channels=['F4'])
    print()