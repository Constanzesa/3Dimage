"""
Object recognition Things-EEG2 dataset

use 250 Hz data
"""

import os
import argparse
import random
import itertools
import datetime
import time
import numpy as np
import pandas as pd
from scipy.signal import resample

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from torch.autograd import Variable
from einops.layers.torch import Rearrange

from sklearn.model_selection import train_test_split

# In the paper the validation set consists in 740 randomly picked trials, we have to adapt this number to our small dataset size
RAND_TRIAL_VAL = 30  # 740 
temp = {}

def sliding_window_data(data, labels, fs, window_length, overlap):
    """
    Segments data into epochs using a sliding window approach.

    Parameters:
    - data: np.array, shape (num_trials, num_channels, num_samples)
    - labels: np.array, shape (num_trials,)
    - fs: int, Sampling frequency (samples per second)
    - window_length: float, Length of each window in seconds
    - overlap: float, Overlap between consecutive windows (in range [0, 1])

    Returns:
    - segmented_data: np.array, shape (num_windows_total, num_channels, samples_per_window)
    - segmented_labels: np.array, shape (num_windows_total,)
    """
    samples_per_window = int(fs * window_length)  # Number of samples per window
    step_size = int(samples_per_window * (1 - overlap))  # Step size based on overlap

    num_trials, num_channels, num_samples = data.shape
    num_windows_per_trial = (num_samples - samples_per_window) // step_size + 1
    num_windows_total = num_trials * num_windows_per_trial

    # Initialize arrays for segmented data and labels
    segmented_data = np.zeros((num_windows_total, num_channels, samples_per_window))
    segmented_labels = np.zeros(num_windows_total)

    window_index = 0
    for trial in range(num_trials):
        for window in range(num_windows_per_trial):
            start_sample = window * step_size
            end_sample = start_sample + samples_per_window
            segmented_data[window_index, :, :] = data[trial, :, start_sample:end_sample]
            segmented_labels[window_index] = labels[trial]
            window_index += 1

    return segmented_data, segmented_labels



import os 

print(os.listdir(".."))

gpus = [1]
result_path = './results-fluid/' 
model_idx = 'fluid-test0-3d'
 
parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--epoch', default='200', type=int)
# One subject only
parser.add_argument('--num_sub', default=1, type=int, #10
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=500, type=int, # 1000
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=2023, type=int,
                    help='seed for initializing training. ')


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            #nn.AvgPool2d((1, 5), (1, 2)), # Wrong dimension
            nn.AvgPool2d((1, 8), (1, 4)),  # 8s window size, stride can change
            #nn.AvgPool2d((1, 51), (1, 5)), # original THINGS
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

        
class Proj_eeg(nn.Sequential):
    #  mat1 and mat2 shapes cannot be multiplied (32x121760 and 31600x768)
    def __init__(self, embedding_dim=161360, proj_dim=768, drop_proj=0.5): # here change dimension 
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
    def forward(self, x):
        return x 

# Image2EEG
class IE():
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 6
        self.labels = []
        self.batch_size = args.batch_size
        self.batch_size_test = 400
        self.batch_size_img = 500 
        self.n_epochs = args.epoch

        self.lambda_cen = 0.003
        self.alpha = 0.5

        self.proj_dim = 256

        self.lr = 0.0002 #
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.start_epoch = 0

        # Adapt the path
        self.eeg_data_path = 'F:/workspace/imagery2024/PREPROCESSED_DATA/3D/' # EEG data '../imagery2024/PREPROCESSED_DATA/data/eeg/S01'
        self.img_data_path = './image_embeddings/' # Image embeddings './dnn_feature/'
        self.test_center_path = './image_embeddings/' # Centroid embeddings
        self.pretrain = False

        self.log_write = open(result_path + "log_subject%d.txt" % self.nSub, "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.Enc_eeg = Enc_eeg().cuda()
        self.Proj_eeg = Proj_eeg().cuda()
        self.Proj_img = Proj_img().cuda()
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.centers = {}
        print('initial define done.')



    def get_eeg_fluid_data(self):

        data = np.load(self.eeg_data_path  + 'P10_01.npy')  # eeg_nofP10_01
        labels = np.load(self.eeg_data_path  + 'group_labelP10_01.npy')


        # 1s window = 1 image embedding
        segmented_data, segmented_labels = sliding_window_data(data, labels, fs=512, window_length=8, overlap=0)

        self.labels = segmented_labels

        # /!\ data is structured as (trial, time, channel) while we want (trial, channel, time)
        data = segmented_data.transpose(0, 2, 1)


        # Resample the data from 512 Hz to 256 Hz
        n_samples = data.shape[-1] // 2  # Divide the number of samples by 2 to downsample
        downsampled_data = resample(data, n_samples, axis=-1)
        data = downsampled_data

        
        X_data = torch.from_numpy(data).type(torch.float)
        X_data = X_data.unsqueeze(1)
        y_labels = torch.from_numpy(segmented_labels).type(torch.LongTensor) - 1


        train_data, test_data, train_label, test_label = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

        print("test_label = ", test_label)

        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        print(f"Train labels shape: {train_label.shape}")
        print(f"Test labels shape: {test_label.shape}")

        print("LABELS", y_labels, len(y_labels))
        print("X LEN ", len(X_data))

        return train_data, train_label, test_data, test_label



    def get_image_data_fluid(self):
        
        data = np.load(self.img_data_path + '3d_' + self.args.dnn + '_feature_maps_training.npy', allow_pickle=True)

        # if window = 8s, compute the mean embedding for each 3d object or use the embedding for front image 2D i.e. clip_feature_maps_training.npy

        data = np.load(self.img_data_path + self.args.dnn + '_feature_maps_training.npy', allow_pickle=True)

        data = np.load(self.img_data_path + 'features_3d_8s.npy', allow_pickle=True)


        
        
        # extend the array copying embedding times 6 (trials) : 624 x 6 = 3744
        data = [element for element in data for _ in range(6)]

        print("NEW SIZE", len(data))#, data.shape)

        labels = self.labels #np.load(self.eeg_data_path  + 'trial_labelP10_01.npy') 


        data = np.squeeze(data) 
        X_data = torch.from_numpy(data).type(torch.float)

        y_labels = torch.from_numpy(labels).type(torch.LongTensor) - 1

        print("X_EMB LEN ", len(X_data), len(labels))

        train_data, test_data, train_label, test_label = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

        print(f"Image feature train data shape: {train_data.shape}")
        print(f"Image feature test data shape: {test_data.shape}")
        print(f"Train labels shape: {train_label.shape}")
        print(f"Test labels shape: {test_label.shape}")

        return train_data, test_data



        
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train(self):
        
        self.Enc_eeg.apply(weights_init_normal)
        self.Proj_eeg.apply(weights_init_normal)
        self.Proj_img.apply(weights_init_normal)

        train_eeg, _, test_eeg, test_label = self.get_eeg_fluid_data()
        train_img_feature, _ = self.get_image_data_fluid()

        test_center = np.load(self.test_center_path + 'centroids.npy', allow_pickle=True) # 'centroid_3d.npy'
        test_center = np.squeeze(test_center)

        print("CENTER", test_center.shape)


        # " We randomly select 740 trials from training data as the validation set in each run of the code "
        val_eeg = train_eeg[:RAND_TRIAL_VAL] # torch.from_numpy(train_eeg[:740])  
        val_image = train_img_feature[:RAND_TRIAL_VAL]

        print("train eeg before", len(train_eeg))

        train_eeg = train_eeg[RAND_TRIAL_VAL:] #torch.from_numpy(train_eeg[740:])

        print("train eeg after", len(train_eeg))
        train_image = train_img_feature[RAND_TRIAL_VAL:]


        dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        print(f"Dataset length: {len(dataset)}")  # Should be > 0

        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        # Already a tensor
        test_eeg = test_eeg #torch.from_numpy(test_eeg)
        test_center = torch.from_numpy(test_center)
        test_label = test_label #torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)

        # Optimizers
        self.optimizer = torch.optim.Adam(itertools.chain(self.Enc_eeg.parameters(), self.Proj_eeg.parameters(), self.Proj_img.parameters()), lr=self.lr, betas=(self.b1, self.b2))

        num = 0
        best_loss_val = np.inf

        for e in range(self.n_epochs):
            in_epoch = time.time()

            self.Enc_eeg.train()
            self.Proj_eeg.train()
            self.Proj_img.train()


            for i, (eeg, img) in enumerate(self.dataloader):

                eeg = Variable(eeg.cuda().type(self.Tensor))
                labels = torch.arange(eeg.shape[0])  # used for the loss
                labels = Variable(labels.cuda().type(self.LongTensor))

                # obtain the features
                eeg_features = self.Enc_eeg(eeg)
                img_features = Variable(img.cuda().type(self.Tensor))
              

                # project the features to a multimodal embedding space
                eeg_features = self.Proj_eeg(eeg_features)  # dim issue here
                img_features = self.Proj_img(img_features)

                # normalize the features
                eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
                img_features = img_features / img_features.norm(dim=1, keepdim=True)

                # cosine similarity as the logits
                logit_scale = self.logit_scale.exp()

                # Squeeze img_feature 1-dim
                img_features = img_features.squeeze(1).squeeze(1)
                logits_per_eeg = logit_scale * eeg_features @ img_features.t()
                logits_per_img = logits_per_eeg.t()

                loss_eeg = self.criterion_cls(logits_per_eeg, labels)
                loss_img = self.criterion_cls(logits_per_img, labels)

                loss_cos = (loss_eeg + loss_img) / 2

                # total loss
                loss = loss_cos

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            if (e + 1) % 1 == 0:
                self.Enc_eeg.eval()
                self.Proj_eeg.eval()
                self.Proj_img.eval()
                with torch.no_grad():
                    # * validation part
                    for i, (veeg, vimg) in enumerate(self.val_dataloader):

                        veeg = Variable(veeg.cuda().type(self.Tensor))
                        vimg_features = Variable(vimg.cuda().type(self.Tensor))
                        vlabels = torch.arange(veeg.shape[0])
                        vlabels = Variable(vlabels.cuda().type(self.LongTensor))

                        veeg_features = self.Enc_eeg(veeg)
                        veeg_features = self.Proj_eeg(veeg_features)
                        vimg_features = self.Proj_img(vimg_features)

                        veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
                        vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

                        logit_scale = self.logit_scale.exp()
                        vimg_features = vimg_features.squeeze(1).squeeze(1) # pb dim
                        vlogits_per_eeg = logit_scale * veeg_features @ vimg_features.t()
                        vlogits_per_img = vlogits_per_eeg.t()

                        vloss_eeg = self.criterion_cls(vlogits_per_eeg, vlabels)
                        vloss_img = self.criterion_cls(vlogits_per_img, vlabels)

                        vloss = (vloss_eeg + vloss_img) / 2

                        if vloss <= best_loss_val:
                            best_loss_val = vloss
                            best_epoch = e + 1
                            torch.save(self.Enc_eeg.module.state_dict(), './model-fluid/' + model_idx + 'Enc_eeg_cls.pth')
                            torch.save(self.Proj_eeg.module.state_dict(), './model-fluid/' + model_idx + 'Proj_eeg_cls.pth')
                            torch.save(self.Proj_img.module.state_dict(), './model-fluid/' + model_idx + 'Proj_img_cls.pth')

                print('Epoch:', e,
                      '  Cos eeg: %.4f' % loss_eeg.detach().cpu().numpy(),
                      '  Cos img: %.4f' % loss_img.detach().cpu().numpy(),
                      '  loss val: %.4f' % vloss.detach().cpu().numpy(),
                      )
                self.log_write.write('Epoch %d: Cos eeg: %.4f, Cos img: %.4f, loss val: %.4f\n'%(e, loss_eeg.detach().cpu().numpy(), loss_img.detach().cpu().numpy(), vloss.detach().cpu().numpy()))
                temp["loss_val"]= vloss.detach().cpu().numpy()


        # * test part
        all_center = test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        self.Enc_eeg.load_state_dict(torch.load('./model-fluid/' + model_idx + 'Enc_eeg_cls.pth'), strict=False)
        self.Proj_eeg.load_state_dict(torch.load('./model-fluid/' + model_idx + 'Proj_eeg_cls.pth'), strict=False)
        self.Proj_img.load_state_dict(torch.load('./model-fluid/' + model_idx + 'Proj_img_cls.pth'), strict=False)

        self.Enc_eeg.eval()
        self.Proj_eeg.eval()
        self.Proj_img.eval()

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = Variable(teeg.type(self.Tensor))
                tlabel = Variable(tlabel.type(self.LongTensor))
                all_center = Variable(all_center.type(self.Tensor))            

                tfea = self.Proj_eeg(self.Enc_eeg(teeg))
                tfea = tfea / tfea.norm(dim=1, keepdim=True)
                similarity = (tfea @ all_center.t()).softmax(dim=-1)  
                _, indices = similarity.topk(5) 

                tt_label = tlabel.view(-1, 1)
                total += tlabel.size(0)
                top1 += (tt_label == indices[:, :1]).sum().item()
                top3 += (tt_label == indices[:, :3]).sum().item()
                top5 += (tt_label == indices).sum().item()

            
            top1_acc = float(top1) / float(total)
            top3_acc = float(top3) / float(total)
            top5_acc = float(top5) / float(total)
        
        print('The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))
        self.log_write.write('The best epoch is: %d\n' % best_epoch)
        self.log_write.write('The test Top1-%.6f, Top3-%.6f, Top5-%.6f\n' % (top1_acc, top3_acc, top5_acc))
        print("ok", temp["loss_val"])
        print("ok2", vloss.detach().cpu().numpy() )
        
        return top1_acc, top3_acc, top5_acc


def main():
    args = parser.parse_args()

    num_sub = args.num_sub   
    cal_num = 0
    aver = []
    aver3 = []
    aver5 = []
    
    for i in range(num_sub):

        cal_num += 1
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(args.seed)

        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print('Subject %d' % (i+1))
        ie = IE(args, i + 1)

        Acc, Acc3, Acc5 = ie.train()
        print('THE BEST ACCURACY IS ' + str(Acc))


        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))

        aver.append(Acc)
        aver3.append(Acc3)
        aver5.append(Acc5)

    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))

    column = np.arange(1, cal_num+1).tolist()
    column.append('ave')
    pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5])
    pd_all.to_csv(result_path + 'result.csv')

if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))