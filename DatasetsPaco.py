import math
import mne
import tensorflow as tf
import torch
import numpy as np
mne.set_log_level(verbose=False)


class EDFData():
    def __init__(self, path, channels=None, binary_labels=True):
        self.path = path
        self.channels = channels if channels else 'all'
        self.binary_labels = binary_labels
        self.epochs, self.sampling_rate = self.get_epochs(path)
        self.id_to_class_dict = {value - 1: key for key, value in self.epochs.event_id.items()}
        self.mean, self.std = self.iterative_mean_std()

    def get_epochs(self, path):
        data = mne.io.read_raw_edf(path)
        sampling_rate = data.info['sfreq']
        events, events_id = mne.events_from_annotations(data, regexp='Sleep stage [A-Z]\d*')
        # events, events_id = mne.events_from_annotations(data, regexp='Sleep stage [N-R]\d*')

        if self.binary_labels:
            events, events_id = self.get_binary_events_eventsids(events, events_id)

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
        for i, a in enumerate(epochs, 0):
            mean = mean + (1 / (i + 1)) * (a.mean(axis=-1) - mean)
        return mean

    def iterative_mean_std(self):
        std = 0
        mean = 0
        for i, a in enumerate(self.epochs, 0):
            a = a.mean(axis=-1)
            new_mean = mean + (1 / (i + 1)) * (a - mean)
            std = std + (1 / (i + 1)) * ((a - mean) * (a - new_mean) - std)
            mean = new_mean
        return mean, std ** (1 / 2)

    @staticmethod
    def get_binary_events_eventsids(events, events_id, awake_label='Sleep stage W'):
        assert awake_label in events_id

        for e in events:
            if e[-1] == events_id[awake_label]:
                e[-1] = 1
            else:
                e[-1] = 2

        for i in events_id:
            if i == awake_label:
                events_id[i] = 1
            else:
                events_id[i] = 2

        return events, events_id


class EDFData_TF_old(EDFData, tf.keras.utils.Sequence):
    def __init__(self, path, batch_size, channels=None, binary_labels=False):
        EDFData.__init__(self, path, channels)
        tf.keras.utils.Sequence.__init__(self)

        self.path = path
        self.batch_size = batch_size
        self.channels = channels if channels else 'all' # => ?
        self.binary_labels = binary_labels
        self.id_to_class_dict = {value: key for key, value in self.epochs.event_id.items()}

    def __getitem__(self, idx):

        X = self.epochs[idx * self.batch_size:(idx + 1) * self.batch_size].get_data()
        Y = self.epochs[idx * self.batch_size:(idx + 1) * self.batch_size].events[:, -1]

        # X = self.epochs.get_data() => senza batch
        # Y = self.epochs.events[:, -1]

        X = (X - self.mean)/self.std

        return np.transpose(X, (0, 2, 1)), Y


    def __len__(self):
        return math.ceil(len(self.epochs) / self.batch_size)


class EDFData_TF(EDFData, tf.keras.utils.Sequence):
    def __init__(self, path, batch_size, channels=None):
        EDFData.__init__(self, path, channels)
        tf.keras.utils.Sequence.__init__(self)
        self.batch_size = batch_size

    def __getitem__(self, idx):
        X = self.epochs[idx * self.batch_size:(idx + 1) * self.batch_size].load_data()._data
        Y = self.epochs[idx * self.batch_size:(idx + 1) * self.batch_size].events[:, -1] - 1
        return X, Y

    def __len__(self):
        return math.ceil(len(self.epochs) / self.batch_size)

class EDFData_GEN_TF(EDFData):
    def __init__(self, path, channels=None):
        EDFData.__init__(self, path, channels)

    def __getitem__(self, idx):
        X = np.transpose(np.squeeze(self.epochs[idx].load_data()._data, axis=0), (1, 0)) # load_data()._data = get_data()
        Y = self.epochs[idx].events[0][-1] - 1 
        X = (X-self.mean)/self.std
        return X, Y

    def __len__(self):
        return len(self.epochs)