import os
import torch
# from .Load_Custom import Spoofing_custom
from .Load_CSV import Spoofing_custom


class dataset_info(object):

    def __init__(self):
        self.root_dir = ""


class data_merge(object):

    def __init__(self, image_dir, train_size=500, val_size=50):
        self.dic = {}
        self.datasets = ["PATCHNET"]
        self.image_dir = image_dir
        PATCHNET = dataset_info()
        PATCHNET.root_dir = os.path.join(self.image_dir, "PATCHNET_DATASET")
        self.dic["PATCHNET"] = PATCHNET
        self.train_size = train_size
        self.test_size = val_size


    def get_single_dataset(self, data_name="", train=True, img_size=256, map_size=32, transform=None, debug_subset_size=None, UUID=-1):
        depth_dir = os.path.join(self.image_dir, 'PATCHNET_DEPTH')
        train_csv = 'datasets/train.csv'
        val_csv = 'datasets/val.csv'
        if train:
            if data_name in self.datasets:
                data_set = Spoofing_custom(train_csv, depth_dir, transform=transform, img_size=img_size, map_size=map_size, UUID=UUID, size=self.train_size)
            else:
                print("NO DATASET Found")
                exit()    
            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        else:
            if data_name in self.datasets:
                data_set = Spoofing_custom(val_csv, depth_dir, transform=transform, img_size=img_size, map_size=map_size, UUID=UUID, size=self.test_size)
            else:
                print("NO DATASET Found")
                exit() 
            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        print("Loading {}, number: {}".format(data_name, len(data_set)))
        return data_set

    def get_datasets(self, train=True, protocol="Patchnet", img_size=256, map_size=32, transform=None, debug_subset_size=None):
        if protocol == "Patchnet":
            data_name_list_train = ["PATCHNET"]
            data_name_list_test = ["PATCHNET"]    
        else:
            print("No such protocol", protocol)
            exit()
        sum_n = 0

        if train:
            data_set_sum = self.get_single_dataset(data_name=data_name_list_train[0], train=True, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=0)
        else:
            data_set_sum = self.get_single_dataset(data_name=data_name_list_test[0], train=False, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=0)

        print("Total number: {}".format(sum_n))
        return data_set_sum
