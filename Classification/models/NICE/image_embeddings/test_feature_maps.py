from transformers import CLIPModel, CLIPTokenizer
import torch
import numpy as np

# Load CLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# Prepare text labels
text_labels = ["banana", "basketball", "faces", "panda", "strawberry", "tiger"]
text_tokens = tokenizer(text_labels, padding=True, return_tensors="pt").to(device)

# Encode text labels
with torch.no_grad():
    text_features = model.get_text_features(**text_tokens)

# Normalize text features
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Load image feature maps
data = np.load('../imagery2024/PREPROCESSED_DATA/2D/DNN_feature_maps/pca_feature_maps/clip/pretrained-True/clip_feature_maps_training.npy', allow_pickle=True)
# Assuming data is in shape (num_samples, num_channels, num_features)
# You may need to reshape or flatten if necessary

# Flatten feature maps if they are 3D (e.g., [num_samples, num_channels, num_features])
data = np.mean(data, axis=1)  # Example: take the mean across the channels
data = torch.tensor(data).float().to(device)

# Normalize image features
img_features = data / data.norm(dim=-1, keepdim=True)

# Compute cosine similarities
similarities = torch.matmul(img_features, text_features.T)

# Find the most similar text labels
most_similar_text_indices = similarities.argmax(dim=-1)
predicted_labels = [text_labels[idx] for idx in most_similar_text_indices]

# Print results
for i, label in enumerate(predicted_labels):
    print(f"Image embedding {i} corresponds to label: {label}")
