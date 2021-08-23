import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt


class History():
    """
    Class designed to track the metrics during the training of a NN.
    """
    
    def __init__(self):
        self.history = {}

    def update(self, history):
        """
        Adds new values to the history.

        Parameters
        ----------
        history: dict{str:float}

        Returns
        -------
        None
        """
        for metric_name, metric_value in history.items():
            if metric_name not in self.history.keys():
                self.history[metric_name] = [metric_value]
            else:
                self.history[metric_name].append(metric_value)

    def return_lasts(self):
        return {name:value[-1] for name, value in self.history.items()}
        
    
    def plot_history(self, figsize=(16,6)):
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
        metrics_names = [a for a in self.history.keys() if a[:3]!='val']
        
        rows = 1
        cols = len(metrics_names)

        plt.figure(figsize=figsize)
        for i,a in enumerate(metrics_names,1):
            plt.subplot(rows,cols,i)
            plt.title(a)
            plt.plot(self.history[a], label="Train")
            plt.plot(self.history["val_"+a], label="Validation")
            plt.legend()

        plt.show()
    
    def update_summary_metrics(self):
        """
        Updates the summary metrics from WandB to show the best value for each.
        Currently it's considering best as higher. Should do something to 
        choose either the higher or the lower depending on the name maybe.
        """
        for metric_name, metric_values in self.history.items():
            best_idx = np.argmax(metric_values)
            best_metric = metric_values[best_idx]
            wandb.run.summary[f"best_{metric_name}"] = best_metric

def train_step(model, optimizer, loss_fn, history, X, Y, metrics=None):
    """
    Trains the model over a batch.

    Parameters
    ----------
    model: nn.Module
        Torch model.
    optimizer:
        Torch optimizer.
    loss_fn:
        Torch loss function.
    X: torch.Tensor
    Y: torch.Tensor
    metrics: dict{str:functions}
        Dict of functions (metrics) we want to calculate and their name.
    
    Returns
    -------
    history: dict{str:float}
        Dictionary of metrics.
    """
    model.train()
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()

    history['loss'] = loss.item()
    if metrics:
        with torch.no_grad():
            for name, metric_fn in metrics.items():
                history[name] = metric_fn(pred, Y)

    optimizer.step()
    return history

def val_step(model, optimizer, loss_fn, history, X, Y, metrics=None):
    """
    Calculates the validation metrics over a batch.

    Parameters
    ----------
    model: nn.Module
        Torch model.
    optimizer:
        Torch optimizer.
    loss_fn:
        Torch loss function.
    X: torch.Tensor
    Y: torch.Tensor
    metrics: dict{str:functions}
        Dict of functions (metrics) we want to calculate and their name.
    
    Returns
    -------
    history: dict{str:float}
        Dictionary of metrics.
    """
    model.eval()
    with torch.no_grad():
        pred = model(X)
        loss = loss_fn(pred, Y)

        history['val_loss'] = loss.item()
        if metrics:
            for name, metric_fn in metrics.items():
                history['val_'+name] = metric_fn(pred, Y)

    return history

def train_fn(device, model, optimizer, loss_fn, trainloader, testloader, epochs, metrics, 
             history=History, checkpoint=False, verbose=True, log_wandb=True):
    
    if log_wandb:
        ## Tell wandb to watch the model
        wandb.watch(model, loss_fn, log="all", log_freq=10)

    history_epoch = history()

    for epoch in range(epochs):
        history_batch = history()
        for batch_i, (X, Y) in enumerate(trainloader, 1):
            X, Y = X.to(device).float(), torch.squeeze(Y).long().to(device)
            history_train = train_step(model, optimizer, loss_fn, {}, X, Y, metrics)
            history_batch.update(history_train)
            
        for batch_i, (X, Y) in enumerate(testloader, 1):
            X, Y = X.to(device).float(), torch.squeeze(Y).long().to(device)
            history_val = val_step(model, optimizer, loss_fn, {}, X, Y, metrics)
            history_batch.update(history_val)
            
        history_epoch.update({name:np.mean(values) for name,values in history_batch.history.items()})
        if verbose:
            print(f"Epoch {epoch+1} -> [Train] (Loss) {history_epoch.history['loss'][-1]:.4f} (Acc) {history_epoch.history['accuracy'][-1]:.4f} | [Val] (Loss) {history_epoch.history['val_loss'][-1]:.4f} (Acc) {history_epoch.history['val_accuracy'][-1]:.4f}")
        
        if checkpoint:
            checkpoint.step(model, history_epoch.history['val_accuracy'][-1])

        if log_wandb:
            ## Log the metrics to WandB
            wandb.log(history_epoch.return_lasts())

    return history_epoch