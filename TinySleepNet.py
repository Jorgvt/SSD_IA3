import tensorflow as tf
import numpy as np

class TinySleepNet(tf.keras.Model):

    def __init__(self, sampling_rate, channels, classes):
        super(TinySleepNet, self).__init__()  # dove la classe parent? call to init of super class is missed
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.classes = classes

        self.conv1d = tf.keras.layers.Conv1D(128, kernel_size=self.sampling_rate // 2, padding='same',
                                             activation='relu',
                                             strides=self.sampling_rate, input_shape=(30, 1))
        self.maxPol1D1 = tf.keras.layers.MaxPooling1D()
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.conv1d2 = tf.keras.layers.Conv1D(128, kernel_size=8, padding='same', strides=1, activation="relu")
        self.conv1d3 = tf.keras.layers.Conv1D(128, kernel_size=8, padding='same', strides=1, activation="relu")
        self.conv1d4 = tf.keras.layers.Conv1D(128, kernel_size=8, padding='same', strides=1, activation="relu")
        self.maxPol1D2 = tf.keras.layers.MaxPooling1D(4)
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.LSTM = tf.keras.layers.LSTM(128)
        self.dense = tf.keras.layers.Dense(self.classes-1, activation="sigmoid")

    def call(self, x):
        # Signature of method 'TinySleepNet.call()' does not match signature of the base method in class 'Model'
        x = tf.transpose(x, (0, 2, 1))
        x = self.conv1d(x)
        # x = self.maxPol1D1(x)
        # x = self.dropout1(x)
        # x = self.conv1d2(x)
        # x = self.conv1d3(x)
        # x = self.conv1d4(x)
        # x = self.maxPol1D2(x)
        # x = self.dropout2(x)
        x = self.LSTM(x)
        return self.dense(x)

