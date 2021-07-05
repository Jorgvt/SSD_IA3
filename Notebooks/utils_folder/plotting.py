import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from .utils import get_labels_and_preds

def plot_labels(y_true, y_pred, label_mapper=None):
    """
    Plots the labels. Useful to inspect transitions between sleep stages.

    Parameters
    ----------
    y_pred: 1D list-like iterable
    y_true: 1D list-like iterable
    label_mapper: dict{int:str}
        dictionary mapping label numbers to sleep stages' names.
    """
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Pred")
    if label_mapper:
        plt.yticks(list(label_mapper.keys()), list(label_mapper.values()))
    plt.legend()
    plt.xlabel("Epochs")

def plot_labels_bundle_pth(device, model, dataset, dataloader):
    """
    Plots the true labels and the predicted labels versus time from a model and the dataset and dataloader.
    """
    ## Get labels and preds
    labels, preds = get_labels_and_preds(device, model, dataloader)
    
    ## Plot the figure
    plot_labels(labels, preds, dataset.id_to_class_dict)

def plot_heatmaps_raw(data, threshold = None, precision = 1, cmap='magma'):    
    """
    Generate a heatmap from a matrix of data (can be a dataframe as well).
    Doesn't include ticks on axis.

    Arguments:
    data -> np.array / pd.DataFrame
    threshold -> float
        Threshold used to change the font color.
    precision -> float
        Float numbers precision.
    Returns:
    None
    """
    
    plt.matshow(data, cmap=cmap, fignum = 0, aspect = "auto")

    if not threshold:
        threshold = ((np.max(data) + np.min(data))/2).mean()
        
    for (i, j), z in np.ndenumerate(data):
        if z < threshold:
            color = 'w'
        else:
            color = 'k'

        plt.text(j, i, f"{z:0.{precision}f}", ha = "center", va = "center", color = color)

def plot_cm(labels, preds, dataset, cmap='magma'):
    """
    Plots the confusion matrix for a set of labels and preds from a dataset.
    """
    plot_heatmaps_raw(confusion_matrix(labels, preds), precision=0, cmap=cmap)
    plt.colorbar()
    plt.xticks(range(len(dataset.id_to_class_dict)), dataset.id_to_class_dict.values(), rotation=90)
    plt.yticks(range(len(dataset.id_to_class_dict)), dataset.id_to_class_dict.values(), rotation=0)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

def plot_cm_bundle_pth(device, model, dataset, dataloader, cmap='magma'):
    """
    Plots the confusion matrix for a model trained with a dataset and dataloader.
    """
    ## Get labels and preds
    labels, preds = get_labels_and_preds(device, model, dataloader)
    
    ## Plot confusion matrix
    plot_cm(labels, preds, dataset, cmap=cmap)

## Probably deprecated and transferred as a method of the History class
def plot_history(history):
    """
    Plots the history values.

    Parameters
    ----------
    history: History object

    Returns
    -------
    None
    """
    ## First retrieve metrics names ## 
    metrics_names = [a for a in history.history.keys() if a[:3]!='val']
    
    rows = 1
    cols = len(metrics_names)

    for i,a in enumerate(metrics_names,1):
        plt.subplot(rows,cols,i)
        plt.title(a)
        plt.plot(history_epoch.history[a], label="Train")
        plt.plot(history_epoch.history["val_"+a], label="Validation")
        plt.legend()