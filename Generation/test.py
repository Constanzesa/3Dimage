#config.eeg_model_ckpt_path = "C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\Generation\\pytorch\\results\\wandb-logs\\EEGNet_Embedding_512_P001_final\\nr5huye5\\checkpoints\\best-model-epoch=00-val_acc=0.00.ckpt"

import torch

# Define the path to your checkpoint file
checkpoint_path = 'C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\Generation\\pytorch\\results\\wandb-logs\\EEGNet_Embedding_512_P001_final\\xz80xf8j\\checkpoints\\best-model-epoch=02-val_acc=0.01.ckpt'

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)

# Check if the checkpoint contains the model state_dict
if 'state_dict' in checkpoint:
    model_state_dict = checkpoint['state_dict']
else:
    model_state_dict = checkpoint

# Print the shape of each parameter in the model's state_dict
for key, value in model_state_dict.items():
    print(f"{key}: {value.shape}")
