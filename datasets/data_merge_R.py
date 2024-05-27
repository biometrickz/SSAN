import os
import torch
from .Load_CSV_R import Spoofing_TrainVal, Spoofing_Test

class dataset_info(object):

    def __init__(self):
        self.root_dir = ""


class data_merge(object):

    def __init__(self, image_dir):
        self.dic = {}
        self.datasets = ["PATCHNET"]
        self.image_dir = image_dir
        PATCHNET = dataset_info()
        PATCHNET.root_dir = os.path.join(self.image_dir, "patchnet")
        self.dic["PATCHNET"] = PATCHNET

# train_list - woxcsmad
# train_list2 - wxcsmad
# train_list3 - wolivexcsmad
# train_list - woxcsmad but w sil youtube

    def get_single_dataset(self, data_name="", type='train', img_size=256, transform=None, debug_subset_size=None, UUID=-1):
        train_csv = os.path.join(self.dic["PATCHNET"].root_dir, 'train', 'train_list4.csv')
        val_csv = os.path.join(self.dic["PATCHNET"].root_dir, 'val', 'val_list.csv')
        test_csv = os.path.join(self.dic["PATCHNET"].root_dir, 'test', 'test_list.csv')
        if type == 'train':
            if data_name in self.datasets:
                data_set = Spoofing_TrainVal(train_csv, transform=transform, img_size=img_size, UUID=UUID)
            else:
                print("NO DATASET Found")
                exit()    
            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        elif type == 'val':
            if data_name in self.datasets:
                print("HEY", data_name)
                data_set = Spoofing_TrainVal(val_csv, transform=transform, img_size=img_size, UUID=UUID)
            else:
                print("NO DATASET Found")
                exit() 
            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        elif type == 'test':
            if data_name in self.datasets:
                print("HEY", data_name)
                data_set = Spoofing_Test(test_csv, transform=transform, img_size=img_size, UUID=UUID)
            else:
                print("NO DATASET Found")
                exit() 
            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        print("Loading {}, number: {}".format(data_name, len(data_set)))
        return data_set

    def get_datasets(self, type='train', protocol="Patchnet", img_size=256, map_size=32, transform=None, debug_subset_size=None):
        if protocol == "Patchnet":
            data_name_list_train = ["PATCHNET"]
            data_name_list_val = ["PATCHNET"]    
            data_name_list_test = ["PATCHNET"]    

        else:
            print("No such protocol", protocol)
            exit()
        data_set_sum = None

        if type == 'train':
            data_set_sum = self.get_single_dataset(data_name=data_name_list_train[0], type=type, img_size=img_size,  transform=transform, debug_subset_size=debug_subset_size, UUID=0)
        elif type == 'val':
            data_set_sum = self.get_single_dataset(data_name=data_name_list_val[0], type=type, img_size=img_size, transform=transform, debug_subset_size=debug_subset_size, UUID=0)
        elif type == 'test':
            data_set_sum = self.get_single_dataset(data_name=data_name_list_test[0], type=type, img_size=img_size, transform=transform, debug_subset_size=debug_subset_size, UUID=0)

        print("Total number: {}".format(len(data_set_sum)))
        return data_set_sum
