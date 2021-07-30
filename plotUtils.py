import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_cm (labels_real, labels_predicted, labels):
    classes_num = np.unique(labels_real).shape[0]
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    cm = confusion_matrix(labels_real, labels_predicted)
    df_cm = pd.DataFrame(cm, columns=labels)
    sn.heatmap(df_cm, yticklabels=labels, annot=True, fmt='g')
    plt.show()

def plot_history(history):

    metrics = [k for k in history.history.keys() if not k.startswith('val_')]
    for m in metrics: 
        # vertical plot
        plt.xlabel('epoch')
        plt.ylabel(m)
        plt.plot(history.history[m])
        plt.plot(history.history['val_'+m])
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

def plot_hypnogram(labels_real, labels_predicted, labels):
    plt.plot(labels_predicted)
    plt.plot(labels_real)
    plt.yticks(list(labels.keys()), list(labels.values()))
    plt.legend(["True", "Predicted"], loc='upper right')
    plt.show()