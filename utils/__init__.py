from .performance import performances_val, performances_val2
import os
import torch.nn.functional as F
import numpy as np


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def extract_name_before_jpg(filename: str, extension='.jpg') -> str:
    # Find the index of the substring '.jpg' in the filename
    index = filename.rfind(extension)
    
    # Extract everything before the '.jpg' ending using string slicing
    if index != -1:
        # Slice the filename to extract the part before the ending
        name_before_jpg = filename[:index]
    else:
        # If '.jpg' is not found, return the whole filename
        name_before_jpg = filename
        
    return name_before_jpg


def get_file_extension(filename: str) -> str:
    # Use os.path.splitext() to split the filename into root and extension
    root, extension = os.path.splitext(filename)
    
    # Return the extension, which includes the dot (e.g., '.jpg')
    return extension