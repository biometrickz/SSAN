import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
import os 
from utils import *
from glob import glob


class Spoofing_TrainVal(Dataset):
    
    def __init__(self, info_list, transform=None, img_size=256, UUID=-1):
        self.labels = pd.read_csv(info_list, delimiter=",", header=None).drop([0], axis=0)
        self.labels[0] = [x.replace("/mnt/8TB/ml_projects_yeldar/patchnet", "/app/data_dir") for x in self.labels[0]]
        self.labels[0] = [x.replace("/mnt/8TB/ml_projects_yeldar/MaskSynthez", "/app/data_dir") for x in self.labels[0]]
        self.labels[0] = [x.replace("/mnt/8TB/ml_projects_yeldar/cropped_youtube_insta_tiktok", "/app/data_dir/mask_test") for x in self.labels[0]]
        self.labels[0] = [x.replace("/app//app", "/app") for x in self.labels[0]]

        self.transform = transform
        self.img_size = img_size
        self.UUID = UUID

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_path =  str(self.labels.iloc[idx, 0])# image_path = image_path.replace('/app//app/', '/app/')
        spoofing_label = int(float(self.labels.iloc[idx, 1]))
        try:
            image_x = self.get_single_image_x(image_path)

            sample = {'image_x': image_x, 'label': spoofing_label, "UUID": self.UUID}
            if self.transform:
                sample = self.transform(sample)
            
            return sample
        except Exception as e:
            print(f"Warning: Could not read image at {image_path}. Skipping this image. Error: {e}")
            return self.__getitem__((idx + 1) % len(self.labels))

    
    def get_single_image_x(self, image_path):
        image_x_temp = cv2.imread(image_path)
        image_x = cv2.resize(image_x_temp, (self.img_size, self.img_size))

        return image_x

class Spoofing_Test(Dataset):
    
    def __init__(self, info_list, transform=None, img_size=256, UUID=-1):
        self.labels = pd.read_csv(info_list, delimiter=",", header=None).drop([0], axis=0)
        self.transform = transform
        self.img_size = img_size
        self.UUID = UUID

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_path =  str(self.labels.iloc[idx, 0])
        spoofing_label = int(float(self.labels.iloc[idx, 1]))
        try:
            image_x_temp = cv2.imread(image_path)
            image_x = cv2.resize(image_x_temp, (self.img_size, self.img_size))
            sample = {'image_x': image_x, 'label': spoofing_label, "UUID": self.UUID}
            if self.transform:
                sample = self.transform(sample)
            
            return sample
        except Exception as e:
            print(f"Warning: Could not read image at {image_path}. Skipping this image. Error: {e}")
            return self.__getitem__((idx + 1) % len(self.labels))
