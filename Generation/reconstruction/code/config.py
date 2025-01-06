import os
import numpy as np

class Config_Generative_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = '.'
        self.roi = 'VC'
        #self.patch_size = 16
        self.pretrain_gm_path = "C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\Generation\\reconstruction\\pretrains\\ldm\\label2img\\config.yaml"
        # self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/semantic')
        # self.pretrain_gm_path = os.path.join(self.root_path, '.\pretrains\ldm\label2img') 
        # self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/text2img-large')
        # self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/layout2img')
        
        self.dataset = 'EEG'
        #self.pretrain_mbm_path = os.path.join(self.root_path, f'reconstruction/pretrains/{self.dataset}/fmri_encoder.pth') 

        self.img_size = 256

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 5 
        self.lr = 5.3e-5
        self.num_epoch = 200
        
        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.global_pool = False
        self.use_time_cond = True
        self.eval_avg = True
        self.clip_tune = True #ADJUSTED


        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None # os.path.join(self.root_path, 'results/generation/25-08-2022-08:02:55/checkpoint.pth')
        
        #Modifications
        self.eeg_data_path_train = "C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\DATA\\S01\\sec_624\\train"
        self.eeg_val_path = "C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\DATA\\S01\\sec_624\\test"
        self.eeg_model_config_path = "C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\Generation\\reconstruction\\pretrains\\EEG\\P001_model_config.yaml"
        # self.eeg_model_ckpt_path = "C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\Generation\\pytorch\\results\\wandb-logs\\EEGNet_Embedding_512_P001_final\\d5ia8swl\\checkpoints\\best-model-epoch=00-val_acc=0.18.ckpt"
        self.eeg_model_ckpt_path ="C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\Generation\\pytorch\\results\\wandb-logs\\lightning_logs\\6jt99ykp\\checkpoints\\best-model-epoch=00-val_acc=0.23.ckpt"
        self.eeg_data_path_test = "C:\\Users\\mituser\\Desktop\\3DReconstruction\\imagery2024\\DATA\\S01\\sec_624\\test\\data.npy"