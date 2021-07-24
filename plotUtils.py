import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_cm (labels_real, labels_predicted):
    # pero asi los cargo en memo => ? <=
    classes_num = np.unique(labels_real).shape[0]
    cm = confusion_matrix(labels_real, labels_predicted)
    df_cm = pd.DataFrame(cm, range(classes_num), range(classes_num))
    plt.figure(figsize=(3,3))
    sn.set(font_scale=1) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g') # font size
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