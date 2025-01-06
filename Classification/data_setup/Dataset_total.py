import numpy as np
from pathlib import Path
from typing import List, Literal
import torch
import os

class Dataset_Large():
    """
    Pytorch Dataset Large

    This expects multiple recordings and selects one recording as validation and the rest as training data.

    Args:
        data_dir: Path to directory containing the data.
            Expects a directory with multiple run-directories. Concatenates all npy files into one dataset.
        label: Whether to use group labels or labels.
        train: Whether to use the training or test set.
            if True, n-1 runs are returned as dataset
            if False (test), only the hold-out set is loaded for validation.
    """
    def __init__(
            self, 
            data_dir: Path, 
            # label_dir:Path,
            label: Literal["group", "label"], 
            train: bool = True, 
            val_run: str = None,
            special: str = None
            ):
        if label not in ["group", "label"]:
            raise ValueError("option must be either 'group' or 'label'")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_names = "group_labels" if label == "group" else "labels"

        self.labels = np.array()
        self.data = np.array()
        
        for session_path in data_dir.rglob('*'):
            if session_path.is_dir() and session_path.name.startswith('S'):
                label_path = os.path.join(session_path, "len_624", "labels.npy")
                data_path = os.path.join(session_path, "len_624", "data.npy")

            # Load the data and labels from the provided paths
            data = np.load(data_path, allow_pickle=True)  # Shape: (trials, channels, samples)
            labels = np.load(label_path, allow_pickle=True)  # Shape: (trials,)
            labels = labels - 1  # Subtract 1 to make labels 0-indexed

            if special is None:
                data = torch.from_numpy(self.data).type(torch.float32).to(self.device)
            labels = torch.from_numpy(self.labels).type(torch.LongTensor).to(self.device)  # Convert labels to long tensor

            self.data = np.concatenate(self.data, data)
            self.labels = np.concatenate(self.labels, labels)
        
            
        print(f"Data shape: {self.data.shape}, Labels shape: {self.labels.shape}")

        # TODO: ADD TRAIN/VAL SPLIT
        """if train:
            label_dir = Path(r"C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\DATA\\S01\\sec_624\\train\\labels.npy")
        else: 
            label_dir = Path(r"C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\DATA\\S01\\sec_624\\val\\labels.npy")"""

        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
  