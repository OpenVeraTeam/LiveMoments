import os
import sys
import time
import yaml
with open("config/inference_config.yml") as f:
    infer_cfg = yaml.safe_load(f)

import numpy as np
import cv2
from tqdm import tqdm
sys.path.append(os.getcwd())
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
    
class Real_ESRGANRef_VAL_Dataset_Steps(Dataset):
    def __init__(self,
                root_dir_path, 
                ref_dir_path,
                ref_lr_dir_path,
                device="cpu",
                ):
        self.device = device
        self.trans = transforms.ToTensor()
        
        self.lr_img_name = []
        self.hr_img_name = []
        self.prompt_name = []
        self.prompt_embeds_name = []
        self.pool_prompt_embeds_name = []
        self.datalen = []
        
        # image data path
        lr_data_path = root_dir_path
        self.ref_data_path = ref_dir_path
        self.ref_lr_data_path = ref_lr_dir_path
        data_file = sorted(os.listdir(lr_data_path))            
        self.lr_img_name = self.lr_img_name + [os.path.join(lr_data_path, file) for file in data_file]

        self.datalen.append(len(self.lr_img_name))
        
        self.prompt_embeds_default = torch.load(infer_cfg["prompt_embeds_path"], map_location=self.device).squeeze()
        self.pooled_prompt_embeds_default = torch.load(infer_cfg["pooled_prompt_embeds_path"], map_location=self.device).squeeze()

        self.img_nums = len(self.lr_img_name)
    
    def __getitem__(self, idx):
        img_name = self.lr_img_name[idx].split("/")[-1]
        lr_img = ImageOps.exif_transpose(Image.open(self.lr_img_name[idx]).convert("RGB"))  # [0,1]
        lr_img = self.trans(lr_img)
        lr_img = lr_img.squeeze() 
        
        prompt_embeds = self.prompt_embeds_default
        pooled_prompt_embeds = self.pooled_prompt_embeds_default
        
        ref_img = ImageOps.exif_transpose(Image.open(os.path.join(self.ref_data_path, img_name)).convert("RGB"))
        ref_img = self.trans(ref_img).squeeze()
        ref_lr_img = ImageOps.exif_transpose(Image.open(os.path.join(self.ref_lr_data_path, img_name)).convert("RGB"))
        ref_lr_img = self.trans(ref_lr_img).squeeze()

        return {
            "img_name": img_name,
            "lr_img": lr_img,
            "ref_img": ref_img,
            "ref_lr_img": ref_lr_img,
            "prompt_embeds_input": prompt_embeds,
            "pooled_prompt_embeds_input": pooled_prompt_embeds,
            }
    
