import os
from glob import glob
import re 
import math
from pathlib import Path 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mne
mne.set_log_level(verbose=False)
import wandb
from wandb.keras import WandbCallback

from Datasets import *

from utils_folder.callbacks import *
from utils_folder.training_pth import *
from utils_folder.plotting import *
from utils_folder.utils import get_labels_and_preds

class TinySleepNet(nn.Module):
    def __init__(self, sampling_rate, channels, classes, input_length=15360):
        super(TinySleepNet, self).__init__()
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.classes = classes
        self.input_shape = (len(channels), input_length)

        self.feature_extraction = nn.Sequential(*[
            nn.Conv1d(in_channels=len(channels), out_channels=128, kernel_size=sampling_rate//2, stride=sampling_rate//4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        ])

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        
        self.classifier = nn.Linear(128*2, classes)

    def forward(self, X):
        X = self.feature_extraction(X)
        # X, _ = self.lstm(X.permute(0,2,1))
        # X = X[:,-1,:]
        X = X.view(X.shape[0],-1)
        X = self.classifier(X)
        return X

    def calculate_flatten_shape(self):
        """
        Makes a forward pass with a dummy tensor to calculate the output shape
        from the feature_extraction block.
        """
        X = torch.ones(size=(1,*self.input_shape))
        with torch.no_grad():
            X = self.feature_extraction(X)
        return math.prod(X.shape)

def create_model(sampling_rate, input_shape, classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(128, kernel_size=sampling_rate//2, strides=sampling_rate//4, input_shape=input_shape, activation="relu"),
        tf.keras.layers.MaxPool1D(8),
        tf.keras.layers.Conv1D(128, kernel_size=8, strides=1, activation="relu"),
        tf.keras.layers.Conv1D(128, kernel_size=8, strides=1, activation="relu"),
        tf.keras.layers.Conv1D(128, kernel_size=8, strides=1, activation="relu"),
        tf.keras.layers.MaxPool1D(4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(classes)
    ])
    return model

def accuracy_fn(Y_pred, Y_true):
    """
    Calculates the accuracy of our model given its predictions and labels.

    Parameters
    ----------
    Y_pred: torch.Tensor
        Raw output from the nn (logits).
    Y_true: torch.Tensor
        Objective labels.
    
    Returns
    -------
    accuracy: float
    """
    Y_pred = torch.softmax(Y_pred, dim=-1)
    Y_pred = Y_pred.argmax(dim=-1)
    accuracy = torch.where(Y_pred==Y_true, 1, 0).sum() / len(Y_true)

    return accuracy.item()

def get_order(file_path):
    """
    Used to order the results from glob, so that the patients are
    properly concatenated.
    """
    match = file_pattern.match(Path(file_path).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

def weights_init(m):
    """
    Initialize the network's weights with a Xavier-Glorot uniform distribution
    to match Keras' implementation.
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

if __name__ == "__main__":
    
    ## Login into WandB
    wandb.login()

    ## Define the configuration parameters
    config = {
        'epochs':50,
        'classes':5,
        'batch_size':64,
        'learning_rate':0.001,
        'channels':['C3','C4','O1','O2','LOC','ROC','CHIN1'],
        'binary':False,
        'metadata': "Testing from script",
        'test_size':0.3,
        'train_test_split_seed':42
    }
    if config["binary"]:
        config["classes"] = 2

    os.environ["CUDA_VISIBLE_DEVICES"]="2"

    with wandb.init(project='test-pth', entity='jorgvt', config=config):
        config = wandb.config

        ## Define the regex to sort the files and obtain the paths list
        file_pattern = re.compile(r'.*?(\d+).*?')
        sorted_files = sorted(glob("/media/usbdisk/data/ProyectoPSG/data/*.edf"), key=get_order)
        ### Remove the 10th patient
        sorted_files_no_10 = [a for a in sorted_files if re.findall(r'\d+', a)[0]!='10']

        ## Load the data
        datasets = [EDFData_PTH(path_glob, channels=config.channels, binary=config.binary) for path_glob in sorted_files_no_10]

        ## Concat all the individual files' datasets
        dataset_concat = torch.utils.data.ConcatDataset(datasets)
        len_dataset_concat = len(dataset_concat)

        # ## Train-Test split of the data
        # test_size = int(len_dataset_concat*config.test_size)
        # train_size = len_dataset_concat - test_size
        # train, test = torch.utils.data.random_split(dataset_concat, [train_size, test_size], 
        #                                             generator=torch.Generator().manual_seed(config.train_test_split_seed))
        
        ## Train-Test split using Pablo's indexes
        ### Load indexes. They have to be either ints or booleans.
        idx_train = np.loadtxt("../indices_train.txt").astype(int)
        idx_test = np.loadtxt("../indices_test.txt").astype(int)
        ### Subset the concatenated dataset
        train = torch.utils.data.Subset(dataset_concat, indices=idx_train)
        test = torch.utils.data.Subset(dataset_concat, indices=idx_test)
        print(f"Using {len_dataset_concat} samples to train: {len(train)} (Train) & {len(test)} (Test).")

        ## Instance the dataloaders
        trainloader = torch.utils.data.DataLoader(train, batch_size = config.batch_size, drop_last=False, shuffle=True)
        testloader = torch.utils.data.DataLoader(test, batch_size = config.batch_size, drop_last=False, shuffle=True)
        sampling_rate = int(datasets[0].sampling_rate)

        ## Define the model ##
        model = create_model(sampling_rate, input_shape=(15360, config.channels), classes=config.classes)
        model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), 
                         optimizer=tf.optimizers.Adam(learning_rate=1e-3), 
                         metrics=["accuracy"])

        ## Train the model ##
        h = model.fit(train, batch_size=config.batch_size, epochs=config.epochs, validation_data=test,
                      callbacks=[WandbCallback(monitor='val_accuracy')])

        ## Get the labels and predictions for the whole datasets
        labels_train, preds_train = get_labels_and_preds(device, model, trainloader)
        labels_test, preds_test = get_labels_and_preds(device, model, testloader)
        
        ## Create and log figures
        ### Train
        plt.figure(figsize=(20,6))
        plot_labels(labels_train, preds_train, label_mapper=datasets[0].id_to_class_dict)
        wandb.log({"Labels_Preds_Plot_Train":wandb.Image(plt)})
        plt.figure(figsize=(8,8))
        plot_cm(labels_train, preds_train, datasets[0])
        wandb.log({"Confusion_Matrix_Train":wandb.Image(plt)})
        ### Test
        plt.figure(figsize=(20,6))
        plot_labels(labels_test, preds_test, label_mapper=datasets[0].id_to_class_dict)
        wandb.log({"Labels_Preds_Plot_Test":wandb.Image(plt)})
        plt.figure(figsize=(8,8))
        plot_cm(labels_test, preds_test, datasets[0])
        wandb.log({"Confusion_Matrix_Test":wandb.Image(plt)})

        ## Update summary metrics
        # wandb.run.summary["best_val_accuracy"] = checkpoint.best_metric
        h.update_summary_metrics()
