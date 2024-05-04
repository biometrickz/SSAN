import os
import torch
from .Load_Custom import Spoofing_custom


class dataset_info(object):

    def __init__(self):
        self.root_dir = ""


class data_merge(object):

    def __init__(self, image_dir):
        self.dic = {}
        self.datasets = ["INSIGHTFACE", "PADDED", "JOINED", "PATCHNET"]
        self.image_dir = image_dir
        PATCHNET = dataset_info()
        PATCHNET.root_dir = os.path.join(self.image_dir, "PATCHNET_DATASET")
        self.dic["PATCHNET"] = PATCHNET
        INSIGHTFACE = dataset_info()
        INSIGHTFACE.root_dir = os.path.join(self.image_dir, "insight")
        self.dic["INSIGHTFACE"] = INSIGHTFACE
        PADDED = dataset_info()
        PADDED.root_dir = os.path.join(self.image_dir, "padded")
        self.dic["PADDED"] = PADDED
        JOINED = dataset_info()
        JOINED.root_dir = os.path.join(self.image_dir, "joined")
        self.dic["JOINED"] = JOINED
        CUSTOM = dataset_info()
        CUSTOM.root_dir = os.path.join(self.image_dir, "sample_dataset")
        self.dic["Custom"] = CUSTOM


    def get_single_dataset(self, data_name="", train=True, img_size=256, map_size=32, transform=None, debug_subset_size=None, UUID=-1):
        data_dir = self.dic[data_name].root_dir
        depth_dir = os.path.join(self.image_dir, 'PATCHNET_DEPTH')
        if train:
            if data_name in self.datasets:
                data_set = Spoofing_custom(os.path.join(data_dir, "train/live/train_list.csv"), os.path.join(data_dir, "train/live"), depth_dir, transform=transform, img_size=img_size, map_size=map_size, UUID=UUID, size=10000)
                data_set += Spoofing_custom(os.path.join(data_dir, "train/mask/train_list.csv"), os.path.join(data_dir, "train/mask"), depth_dir, transform=transform, img_size=img_size, map_size=map_size, UUID=UUID, size = 10000)
            else:
                print("NO DATASET Found")
                exit()    
            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        else:
            if data_name in self.datasets:
                data_set = Spoofing_custom(os.path.join(data_dir, "val/live/val_list.csv"), os.path.join(data_dir, "val/live"), depth_dir, transform=transform, img_size=img_size, map_size=map_size, UUID=UUID, size = 1000)
                data_set += Spoofing_custom(os.path.join(data_dir, "val/mask/val_list.csv"), os.path.join(data_dir, "val/mask"), depth_dir, transform=transform, img_size=img_size, map_size=map_size, UUID=UUID, size = 1000)
            else:
                print("NO DATASET Found")
                exit() 
            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        print("Loading {}, number: {}".format(data_name, len(data_set)))
        return data_set

    def get_datasets(self, train=True, protocol="Custom", img_size=256, map_size=32, transform=None, debug_subset_size=None):
        if protocol == "Custom":
            data_name_list_train = ["INSIGHTFACE", "PADDED", "JOINED"]
            data_name_list_test = ["INSIGHTFACE", "PADDED", "JOINED"]
        elif protocol == "Patchnet":
            data_name_list_train = ["PATCHNET"]
            data_name_list_test = ["PATCHNET"]    
        else:
            print("No such protocol", protocol)
            exit()
        sum_n = 0
        if train:
            data_set_sum = self.get_single_dataset(data_name=data_name_list_train[0], train=True, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=0)
            sum_n = len(data_set_sum)
            for i in range(1, len(data_name_list_train)):
                data_tmp = self.get_single_dataset(data_name=data_name_list_train[i], train=True, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=i)
                data_set_sum += data_tmp
                sum_n += len(data_tmp)
        else:
            data_set_sum = {}
            for i in range(len(data_name_list_test)):
                data_tmp = self.get_single_dataset(data_name=data_name_list_test[i], train=False, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=i)
                data_set_sum[data_name_list_test[i]] = data_tmp
                sum_n += len(data_tmp)
        print("Total number: {}".format(sum_n))
        return data_set_sum
