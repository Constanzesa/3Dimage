"""
Package CLIP features for center images

"""

# You may average the center of each concept to get the center_xxx.npy.

import argparse
import torch.nn as nn
import numpy as np
import torch
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

gpus = [1]

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--project_dir', default='../imagery2024/PREPROCESSED_DATA/2D/', type=str)
args = parser.parse_args()

print('Extract feature maps CLIP of images for center <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.use_deterministic_algorithms(True)

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model = model.cuda()
model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])


processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

img_set_dir = os.path.join(args.project_dir, '2Dimages/') # 'image_set/center_images/'
condition_list = os.listdir(img_set_dir)
condition_list.sort()

all_centers = []
cond_center = []

# Image directories
img_set_dir = os.path.join(args.project_dir, '2Dimages') # '2Dimages'
img_partitions = os.listdir(img_set_dir)
for p in img_partitions:
    part_dir = os.path.join(img_set_dir, p)
    image_list = []
    for root, dirs, files in os.walk(part_dir):
        for file in files:
            if file.endswith(".png") or file.endswith(".PNG")  or file.endswith(".jpg") or file.endswith(".JPEG"):
                image_list.append(os.path.join(root,file))
    image_list.sort()
    # Create the saving directory if not existing
    save_dir = os.path.join(args.project_dir, 'DNN_feature_maps',
        'full_feature_maps', 'clip', 'pretrained-'+str(args.pretrained), p)
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)

    # Extract and save the feature maps
    # * better to use a dataloader
    for i, image in enumerate(image_list):
        img = Image.open(image).convert('RGB')
        inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=img, return_tensors="pt", padding=True)
        #print("input ", inputs)
        inputs.data['pixel_values'].cuda()
        with torch.no_grad():
            outputs = model(**inputs).image_embeds
            cond_center.append(outputs.detach().cpu().numpy())
            #print(cond_center)
    print(len(cond_center))
    cond_center = np.mean(cond_center, axis=0)
    print("len 2", len(cond_center))
    all_centers.append(np.squeeze(cond_center))
    print("len all ", len(all_centers))

        #file_name = p + '_' + format(i+1, '07')
        #np.save(os.path.join(save_dir, file_name), feats)
print("END len all ", len(all_centers))

all_centers = np.array(all_centers)
print(all_centers.shape)

#np.save(os.path.join(args.project_dir, 'center_fluid_image_clip.npy'), all_centers)



'''
for cond in condition_list:
    one_cond_dir = os.path.join(args.project_dir, '2Dimages/', cond) # 'image_set/center_images/'
    cond_img_list = os.listdir(one_cond_dir)
    cond_img_list.sort()
    cond_center = []
    for img in cond_img_list:
        print("image ", img)
        img_path = os.path.join(one_cond_dir, img)
        img = Image.open(img_path).convert('RGB')
        inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=img, return_tensors="pt", padding=True)
        inputs.data['pixel_values'].cuda()
        with torch.no_grad():
            outputs = model(**inputs).image_embeds
    # * for mean center
    #     cond_center.append(outputs.detach().cpu().numpy())
    # cond_center = np.mean(cond_center, axis=0)
    # all_centers.append(np.squeeze(cond_center))
        cond_center.append(np.squeeze(outputs.detach().cpu().numpy()))
    all_centers.append(np.array(cond_center))


# all_centers = np.array(all_centers)
# print(all_centers.shape)
np.save(os.path.join(args.project_dir, 'center_all_image_clip.npy'), all_centers)


'''