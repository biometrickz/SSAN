import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
import os 
from utils import *
from glob import glob


class Spoofing_custom(Dataset):
    
    def __init__(self, info_list, root_dir,  depth_dir, transform=None, img_size=256, map_size=32, UUID=-1, size=100):
        self.labels = pd.read_csv(info_list, delimiter=",", header=None).drop([0], axis=0)[:size]
        self.root_dir = root_dir
        self.map_root_dir = depth_dir
        self.transform = transform
        self.img_size = img_size
        self.map_size = map_size
        self.UUID = UUID

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_name = str(self.labels.iloc[idx, 1])
        image_path = os.path.join(self.root_dir, image_name)
        spoofing_label = int(self.labels.iloc[idx, 0])
        try:
            image_x, map_x = self.get_single_image_x(image_path, image_name, spoofing_label)

            sample = {'image_x': image_x, 'map_x': map_x, 'label': spoofing_label, "UUID": self.UUID}
            if self.transform:
                sample = self.transform(sample)
            
            return sample
        except Exception as e:
            print(f"Warning: Could not read image at {image_path}. Skipping this image. Error: {e}")
            return self.__getitem__((idx + 1) % len(self.labels))

    
    def get_single_image_x(self, image_path, image_name, spoofing_label):
        
        image_x_temp = cv2.imread(image_path)
        image_x = cv2.resize(image_x_temp, (self.img_size, self.img_size))
        map_x = np.zeros((self.map_size, self.map_size))
        if spoofing_label == 1:
            map_name = "{}_depth.jpg".format(extract_name_without_extension(image_name))
            map_path = os.path.join(self.map_root_dir, map_name)
            try:
                map_x_temp = cv2.imread(map_path, 0)
                map_x = cv2.resize(map_x_temp, (self.map_size, self.map_size))
            except:
                print(map_name)

        return image_x, map_x
