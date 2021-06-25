import matplotlib.pyplot as plt
import numpy as np

def plot_labels(y_pred, y_true, label_mapper=None):
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