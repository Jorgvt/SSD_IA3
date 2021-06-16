import math

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

import Notebooks.DatasetsJorge

if __name__ == '__main__':

    dataset = Notebooks.DatasetsJorge.EDFData_TF("../SSD_IA3/Data/PSG1.edf", batch_size=8, channels=['F4'])
    dataset_2 = Notebooks.DatasetsJorge.EDFData_TF("../SSD_IA3/Data/PSG2.edf", batch_size=8, channels=['F4'])

    # TODO data = np.squeeze(dataset.epochs.load_data()), if not is data = array(None, dtype=object), however:
    #   tensorflow.python.framework.errors_impl.InvalidArgumentError:  Conv2DCustomBackpropInputOp only supports NHWC.
    # 	 [[node gradient_tape/sequential/conv1d_3/conv1d/Conv2DBackpropInput
    data = np.squeeze(dataset.epochs._data)
    print(data.shape)

    # TODO data_2 = np.squeeze(dataset_2.epochs.load_data()), if not is null too
    data_2 = np.squeeze(dataset_2.epochs._data)
    print(data_2.shape)

    labels = []
    for a, b in dataset:
        labels.extend(b)
    labels = np.array(labels)
    print(labels.shape)

    labels_2 = []
    for a, b in dataset_2:
        labels_2.extend(b)
    labels_2 = np.array(labels_2)
    print(labels_2.shape)

    sr = int(dataset.sampling_rate)
    print(sr)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(128, kernel_size=sr // 2, padding='same', strides=sr // 4, activation="relu",
                               data_format='channels_first',
                               input_shape=(len(dataset.channels), 15360)),
        tf.keras.layers.MaxPooling1D(8),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv1D(128, kernel_size=8, padding='same', strides=1, activation="relu",
                               data_format='channels_first'),
        tf.keras.layers.Conv1D(128, kernel_size=8, padding='same', strides=1, activation="relu",
                               data_format='channels_first'),
        tf.keras.layers.Conv1D(128, kernel_size=8, padding='same', strides=1, activation="relu",
                               data_format='channels_first'),
        tf.keras.layers.MaxPooling1D(4),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation="softmax")
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])
    model.summary()

    history = model.fit(np.expand_dims(data, 1), labels, epochs=50, validation_data=(np.expand_dims(data_2, 1), labels_2))

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.legend()
    plt.show()