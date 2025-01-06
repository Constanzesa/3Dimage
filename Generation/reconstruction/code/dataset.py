
import numpy as np
import os
from einops import rearrange
import torch
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import List
from torch.utils.data import random_split, Subset
from torch.utils.data import TensorDataset
from transformers import AutoProcessor


from typing import Callable
from PIL import Image
import pandas as pd

def identity(x):
    return x
def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

def pad_to_length(x, length):
    assert x.ndim == 3
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')

def normalize(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def img_norm(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = (img / 255.0) * 2.0 - 1.0 # to -1 ~ 1
    return img

def channel_first(img):
        if img.shape[-1] == 3:
            return rearrange(img, 'h w c -> c h w')
        return img
    
class EEGDataset():
    """
    EEG: the (preprocessed) EEG files are stored as .npy files (e.g.: ./data/preprocessed/wet/P00x/run_id/data.npy)
    with the individual image labels in the same directory (.../labels.npy) going from 0-599.
    The image labels can be mapped to the image paths via the experiment file used to run Psychopy which is found at
    ./psychopy/loopTemplate1.xlsx

    Images: the images for each image_class are stored in the directory ./data/images/experiment_subset_easy

    Args:
        data_dir: Path to directory containing the data.
            Expects a directory with multiple run-directories. Concatenates all npy files into one dataset.
        image_transform: Optional transform to be applied on images.
        train: Whether to use the training or validation set.
            if True, n-1 runs are returned as dataset
            if False, only the hold-out set is loaded for validation.
        val_run: Name of the run to be used as hold-out set for validation.
            Note: this allows to use a validation set from the same directory or from a different directory by specifying
            a new data_dir for the validation set.
        preload_images: Whether to pre-load all images into memory (speed vs memory trade-off)
    """
    def __init__(
            self, 
            data_dir: Path, 
            image_transform: Callable = None,
            train: bool = True, 
            val_run: str = None,
            preload_images: bool = True,
            ):
        self.preload_images = preload_images
        # self.image_paths = np.array(pd.read_excel("C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\Image3DObjects\\3Dimages_Windows.xlsx", engine='openpyxl').images.values)
           # Read the Excel file containing image paths
        df = pd.read_excel("C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\Image3DObjects\\3Dimages_Windows_new.xlsx", engine='openpyxl')
        # print(f"EXCEL FILE: {df.shape}\n", df)
        
        # self.image_paths = np.array(pd.read_excel("C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\Image3DObjects\\3Dimages_Windows_new.xlsx", engine='openpyxl'))
        base_path = "C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\Image3DObjects"
        
        # # # Correct path concatenation - handle if the path is relative or absolute
        self.image_paths = np.array([
            os.path.join(base_path, path) if not os.path.isabs(path) else path 
            for path in df['images'].values
        ])


       # self.image_paths = np.array([ os.path.join(base_path, img) for img in df['images'].values])

        

       
        self.image_transform = image_transform
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # if all(p.is_dir() for p in Path(data_dir).iterdir()): #data_dir is a directory with multiple run-directories
        if train:
            self.labels = []
            self.data = []
            self.data = np.load("C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\DATA\\S01\\sec_624\\train\\data.npy")
            self.labels = np.load("C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\DATA\\S01\\sec_624\\train\\labels.npy")
        else:
            #NOTE: REPLACE THIS WHEN EVERYTHING RUNS [::10]
            self.data = np.load("C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\DATA\\S01\\sec_624\\val\\data.npy")
            self.labels = np.load("C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\DATA\\S01\\sec_624\\val\\labels.npy")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Shuffle data and labels with the same permutation
        perm = np.random.permutation(len(self.labels))
        self.data = self.data[perm]
        self.labels = self.labels[perm]
        # Load the data and labels from the provided paths
        # self.data = np.load(data_dir, allow_pickle=True)  # Shape: (trials, channels, samples)
        # self.labels = np.load(data_dir, allow_pickle=True)  # Shape: (trials,)
        self.data = torch.from_numpy(self.data).type(torch.float32).to(self.device)
        self.labels = torch.from_numpy(self.labels).type(torch.LongTensor).to(self.device)  # Convert labels to long tensor
        self.labels = self.labels - 1  # Subtract 1 to make labels 0-indexed

       
        # print("LABELS: ", self.labels)
        # if train:
        #     self.labels =[]
        #     self.data = []
        #     self.data = Subset(torch.utils.data.TensorDataset(torch.from_numpy(self.data), torch.from_numpy(self.labels)), 0.75)
        # else:
        #     self.data = Subset(torch.utils.data.TensorDataset(torch.from_numpy(self.data), torch.from_numpy(self.labels)), 0.25)

            
        # self.data = torch.from_numpy(self.data).to(self.device) #swap axes to get (n_trials, channels, samples) 

        # self.data = torch.from_numpy(self.data.swapaxes(1,2)).to(self.device) #swap axes to get (n_trials, channels, samples) 
        # self.labels = torch.from_numpy(self.labels).long().to(self.device) #turn into one-hot encoding torch.nn.functional.one_hot( , num_classes = -1)
        # Create a mapping from each group label to image indices
        self.group_labels = df['group_label'].values        
        self.group_to_indices = {label: np.where(self.group_labels == label)[0] for label in np.unique(self.group_labels)}
        # self.group_to_indices = torch.from_numpy(self.group_to_indices).to(self.device)


        # if self.preload_images:
        #     self.images = np.stack([self.image_transform(np.array(Image.open(path[1:]).convert("RGB"))/255.0) for path in self.image_paths])
        #     self.images = torch.from_numpy(self.images).to(self.device)
        if self.preload_images:
            self.images = np.stack([self.image_transform(np.array(Image.open(path).convert("RGB"))/255.0) for path in self.image_paths])
            self.images = torch.from_numpy(self.images).to(self.device)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # if self.preload_images:
        #     img_idx = self.labels[idx]
        #     return {'eeg': self.data[idx].float(), 'image': self.images[img_idx]}
        # else:
        #     image_raw = Image.open(self.image_paths[self.labels[idx]][1:]).convert("RGB")
        #     image = np.array(image_raw) / 255.0
        #     return {'eeg': self.data[idx].float(), 'image': self.image_transform(image)}
        group_label = self.labels[idx].item()
        image_idx = np.random.choice(self.group_to_indices[group_label])
        if self.preload_images:
            # img_idx = self.labels[idx] ß
            return {'eeg': self.data[idx].float(), 'image': self.images[image_idx]}
            

        # if self.preload_images:s
        #     img_idx = self.labels[idx]
        #     return {'eeg': self.data[idx].float(), 'image': self.images[img_idx]}
        else:
            # Lazy load the image when needed
            image_path = self.image_paths[self.labels[idx]]
            # print("LAZY IMG PATH:", self.image_paths)
            if os.path.exists(image_path):
                image_raw = Image.open(image_path).convert("RGB")
                image = np.array(image_raw) / 255.0
                return {'eeg': self.data[idx].float(), 'image': self.image_transform(image)}
            else:
                raise OSError(f"File not found: {image_path}")