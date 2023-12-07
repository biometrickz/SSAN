import os
import torch
import pandas as pd
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import math
import os 
from utils import *
from glob import glob
import pickle


def crop_face_from_scene(image, bbox, scale):

    #  postprocess bounding boxes
    x1, y1, w, h = bbox
    y2=y1+w
    x2=x1+h
    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)
    region=image[x1:x2,y1:y2]
    
    return region


class Spoofing_custom(Dataset):
    
    def __init__(self, info_list, root_dir,  transform=None, scale_up=1.5, scale_down=1.0, img_size=256, map_size=32, UUID=-1):
        self.labels = pd.read_csv(info_list, delimiter=",", header=None).drop([0], axis=1)
        self.root_dir = root_dir
        self.map_root_dir = os.path.join(root_dir, "depth")
        self.transform = transform
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.img_size = img_size
        self.map_size = map_size
        self.UUID = UUID

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_name = str(self.labels.iloc[idx, 1])
        image_path = os.path.join(self.root_dir, image_name)
        spoofing_label = self.labels.iloc[idx, 0]
        if spoofing_label == 1:
            spoofing_label = 1
        else:
            spoofing_label = 0
        image_x, map_x = self.get_single_image_x(image_path, image_name, spoofing_label)
        sample = {'image_x': image_x, 'map_x': map_x, 'label': spoofing_label, "UUID": self.UUID}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def get_single_image_x(self, image_path, image_name, spoofing_label):
        
        face_scale = np.random.randint(int(self.scale_down*10), int(self.scale_up*10))
        face_scale = face_scale/10.0
        image_x_temp = cv2.imread(image_path)

        
        f_path = os.path.join(self.root_dir, "bboxes.pickle")
        with open(f_path, 'rb') as handle:
            bboxes = pickle.load(handle)

        x, y, w, h = bboxes[image_name]
        print(image_name, bboxes[image_name])
        image_x = cv2.resize(crop_face_from_scene(image_x_temp, [y, x, w, h], face_scale), (self.img_size, self.img_size))
        
        if spoofing_label == 1:
            map_name = "{}_depth.jpeg".format(image_name.split('.')[0])
            map_path = os.path.join(self.map_root_dir, map_name)
            map_x_temp = cv2.imread(map_path, 0)
            try:
                map_x = cv2.resize(crop_face_from_scene(map_x_temp, [y, x, w, h], face_scale), (self.map_size, self.map_size))
            except:
                print(map_name)
        else:
            map_x = np.zeros((self.map_size, self.map_size))

        return image_x, map_x
