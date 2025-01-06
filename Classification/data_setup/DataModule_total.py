import numpy as np
import os
from pathlib import Path
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import List
import random
from sklearn.model_selection import train_test_split
from data_setup.Dataset_total import Dataset_Large

class DataModule(pl.LightningDataModule):
    """
    The DataModule class allows to built dataset agnostic models as it takes care of all the
    data related stuff. It also allows to easily switch between different datasets.

    Args:
        data_dir: Path to directory containing the data.
            Expects either a directory with multiple run-directories or with .npy files.
            - If multiple sub-directories are found, each sub-directory is considered a run.
            and all npy files are concatenated into one dataset.
            - If no sub-directories are found, a dataset is constructed from the npy files.
        val_run: string specifying the validation run for the large dataset (if None, expects Small Dataset)
        batch_size: Batch size for training and validation.
        num_workers: Number of workers for the dataloader.
        seed: Seed for the stratified random split.

    Example:
    The DataModule can be used to setup the model:
        dm = DataModule(...)
        # Init model from datamodule's attributes
        model = Model(*dm.dims, dm.num_classes)

    The DataModule can then be passed to trainer.fit(model, DataModule) to override model hooks.
    """
    def __init__(
            self, 
            data_dir: str, 
            test_dir: str = None,
            val_run: str = None,
            test_run: str = None,
            batch_size: int = 32, 
            num_workers: int = 0, 
            seed: int = 42, 
            special = None,
            **kwargs):
        
        super().__init__()
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.val_run = val_run #val_run is treated here as a directory now 
        self.test_run = test_run # test run takes P10test in 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.special = special #allows to use grayscale_images or fourier_spectrograms instead of data.npy
       
    def setup(self, stage=None):
        #Loads data in from file and prepares PyTorch tensor datsets for each split.
        #If you don't mind loading all datasets at once, stage=None will load both, train (+val) and test.
        # self.data = Dataset_Large(Path(self.data_dir), label="label", train=True, special=self.special)

        if stage == "fit" or stage is None:
            # Directly load train and validation datasets from their respective directories
            self.train_dataset = Dataset_Large(Path(self.data_dir), label="label", split="train", special=self.special)
            self.val_dataset = Dataset_Large(Path(self.val_run), label="label", split="val", special=self.special)

        if stage == "test" or stage is None:
            if self.test_dir:
                self.test_dataset = Dataset_Large(Path(self.test_dir), label="label", split="test", special=self.special)

    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle = True) #, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle = False) #, sampler=self.val_sampler)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        print("Not implemented yet")
        pass
