import torch

class Checkpoint():
    """Class that saves a model whenever a designated metric is improved."""
    
    def __init__(self, path, monitor, mode='max', verbose=True):
        """
        path: str
            Path to save the model.
        monitor: str
            Name of the metric we want to monitor.
            Has to be the same name as in the History object.
        mode: str
            Whether the best metric is the max or the min.
        """
        assert mode in ['max', 'min'], 'mode must be either "max" or "min"'

        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_metric = -999999 if self.mode=='max' else 999999
        self.best_model = None

    def step(self, model, metric):
        if self.mode == 'max':
            if metric > self.best_metric:
                if self.verbose:
                    print(f"{self.monitor} improved from {self.best_metric:.4f} -> {metric:.4f}. Model saved!.")

                self.best_metric = metric
                self.best_model = model
                torch.save(model.state_dict(), self.path)

        elif self.mode == 'min':
            if metric < self.best_metric:
                if self.verbose:
                    print(f"{self.monitor} improved from {self.best_metric:.4f} -> {metric:.4f}. Model saved!.")

                self.best_metric = metric
                self.best_model = model
                torch.save(model.state_dict(), self.path)