import matplotlib.pyplot as plt

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