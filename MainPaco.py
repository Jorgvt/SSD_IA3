import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import Notebooks.DatasetsPaco

if __name__ == '__main__':

    dataset = Notebooks.DatasetsPaco.EDFData_TF_old("../Data/PSG1.edf", batch_size=8, channels=['F4'], binary_labels=False)
    dataset_2 = Notebooks.DatasetsPaco.EDFData_TF_old("../Data/PSG2.edf", batch_size=8, channels=['F4'], binary_labels=False)

    data = np.squeeze(dataset.epochs.load_data()) # np.squeeze(dataset.epochs.load_data()) / np.squeeze(dataset.epochs._data) / dataset.epochs.get_data() / np.squeeze(dataset.epochs._data)
    print(data.shape)
    data_2 = np.squeeze(dataset_2.epochs.load_data())
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

    std = StandardScaler()
    std.fit(data)
    data_std = std.transform(data)
    data_2_std = std.transform(data_2)

    # model, che andrebbe messo da un'altra parte ancora per pulizia
    sr = int(dataset.sampling_rate) # che poi da un getter ancora *
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(128, kernel_size=sr // 2, padding='same', strides=sr // 4, activation="relu",
                               data_format='channels_last',
                               input_shape=(15360, 1)),  # 15360 / 128 = 120
        tf.keras.layers.MaxPooling1D(8),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv1D(128, kernel_size=8, padding='same', strides=1, activation="relu",
                               data_format='channels_last'),
        tf.keras.layers.Conv1D(128, kernel_size=8, padding='same', strides=1, activation="relu",
                               data_format='channels_last'),
        tf.keras.layers.Conv1D(128, kernel_size=8, padding='same', strides=1, activation="relu",
                               data_format='channels_last'),
        tf.keras.layers.MaxPooling1D(4),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation="softmax")
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), # CategoricalCrossentropy()
                  optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])
    model.summary()

    X_tr = np.transpose(np.expand_dims(data, 1), (0, 2, 1))  # samples x # points x # channels, pero no puedo mas si mas que uno eh
    X_te = np.transpose(np.expand_dims(data_2, 1), (0, 2, 1))
    history = model.fit(X_tr, labels-1, epochs=50, validation_data=(X_te, labels_2))

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