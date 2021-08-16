import os
from glob import glob
import re 
import math
from pathlib import Path 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_addons as tfa
import mne
mne.set_log_level(verbose=False)
import wandb
from wandb.keras import WandbCallback

from Datasets import *

from utils_folder.callbacks import *
from utils_folder.training_pth import *
from utils_folder.plotting import *
from utils_folder.utils import get_labels_and_preds

def create_model(sampling_rate, input_shape, classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(128, kernel_size=sampling_rate//2, padding="same", strides=sampling_rate//4, input_shape=input_shape, activation="relu"),
        tf.keras.layers.MaxPool1D(8),
        tf.keras.layers.Conv1D(128, kernel_size=8, padding="same",  strides=1, activation="relu"),
        tf.keras.layers.Conv1D(128, kernel_size=8, padding="same",  strides=1, activation="relu"),
        tf.keras.layers.Conv1D(128, kernel_size=8, padding="same",  strides=1, activation="relu"),
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

def load_labels(glob_path):
    """
    Loads all the labels from the .csv files.
    """
    sorted_files = sorted(glob(glob_path), key=get_order)
    sorted_files_no_10 = [a for a in sorted_files if re.findall(r'\d+', a)[0]!='10']
    labels_npy = [pd.read_csv(a).to_numpy().squeeze() for a in sorted_files_no_10]
    labels_npy = np.concatenate(labels_npy)
    
    return labels_npy


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
        'sampling_rate':512,
    }
    if config["binary"]:
        config["classes"] = 2

    os.environ["CUDA_VISIBLE_DEVICES"]="2"

    with wandb.init(project='test-tf', entity='jorgvt', config=config):
        config = wandb.config

        ## Define the regex to sort the files and obtain the paths list
        file_pattern = re.compile(r'.*?(\d+).*?')
        sorted_files = sorted(glob("/home/pabloro/master/*.npy"), key=get_order)
        ### Remove the 10th patient
        sorted_files_no_10 = [a for a in sorted_files if re.findall(r'\d+', a)[0]!='10']

        ## Load the data
        datos_npy = [np.load(a) for a in sorted_files_no_10]

        ## Concat all the individual files' datasets
        datos_npy = np.concatenate(datos_npy)
        len_dataset = len(datos_npy)

        ## Load the labels
        labels = load_labels("/home/pabloro/master/*.csv")
        ### Instantiate the LabelEncoder
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        idx_to_label = {i:c for i, c in enumerate(label_encoder.classes_)}

        # ## Train-Test split of the data
        # test_size = int(len_dataset_concat*config.test_size)
        # train_size = len_dataset_concat - test_size
        # train, test = torch.utils.data.random_split(dataset_concat, [train_size, test_size], 
        #                                             generator=torch.Generator().manual_seed(config.train_test_split_seed))
        
        ## Train-Val-Test split using Pablo's indexes
        ### Load indexes. They have to be either ints or booleans.
        idx_train = np.loadtxt("../indices_train.txt").astype(int)
        idx_test = np.loadtxt("../indices_test.txt").astype(int)
        idx_s = idx_test.copy()
        np.random.shuffle(idx_s)
        idx_val = idx_s[:len(idx_test)//2]
        idx_test_2 = idx_s[len(idx_test)//2:]
        ### Subset the concatenated dataset
        train = datos_npy[idx_train]
        val = datos_npy[idx_val]
        test = datos_npy[idx_test_2]
        train_labels = labels[idx_train]
        val_labels = labels[idx_val]
        test_labels = labels[idx_test_2]

        ## Create tf.data.Dataset(s) from the npy arrays. Shuffle and batch them.
        train = tf.data.Dataset.from_tensor_slices((train, train_labels)).shuffle(16).batch(config.batch_size)
        val = tf.data.Dataset.from_tensor_slices((val, val_labels)).shuffle(16).batch(config.batch_size)
        test = tf.data.Dataset.from_tensor_slices((test, test_labels)).shuffle(16).batch(config.batch_size)

        print(f"Using {len_dataset} samples to train: {len(train)} (Train) & {len(val)} (Validation) & {len(test)} (Test).")

        ## Define the model ##
        model = create_model(config.sampling_rate, input_shape=(15360, 9), classes=config.classes)
        model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), 
                         optimizer=tf.optimizers.Adam(learning_rate=1e-3), 
                         metrics=["accuracy"])

        ## Train the model ##
        h = model.fit(train, batch_size=config.batch_size, epochs=config.epochs, validation_data=val,
                      callbacks=[WandbCallback(monitor='val_accuracy')])

        ## Load the best performant model
        model.load_weights(os.path.join(wandb.run.dir, "model-best.h5"))

        ## Get the predictions for the whole datasets
        preds_train = model.predict(train).argmax(axis=-1)
        preds_val = model.predict(val).argmax(axis=-1)
        preds_test = model.predict(test).argmax(axis=-1)
        
        ## Create and log figures
        ### Train
        plt.figure(figsize=(20,6))
        plot_labels(train_labels, preds_train, label_mapper=idx_to_label)
        wandb.log({"Labels_Preds_Plot_Train":wandb.Image(plt)})
        plt.figure(figsize=(8,8))
        plot_cm_dict(train_labels, preds_train, label_mapper=idx_to_label)
        wandb.log({"Confusion_Matrix_Train":wandb.Image(plt)})
        ### Test
        plt.figure(figsize=(20,6))
        plot_labels(test_labels, preds_test, label_mapper=idx_to_label)
        wandb.log({"Labels_Preds_Plot_Test":wandb.Image(plt)})
        plt.figure(figsize=(8,8))
        plot_cm_dict(test_labels, preds_test, label_mapper=idx_to_label)
        wandb.log({"Confusion_Matrix_Test":wandb.Image(plt)})
