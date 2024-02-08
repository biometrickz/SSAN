import os
import torch
from .Load_Custom import Spoofing_custom


class dataset_info(object):

    def __init__(self):
        self.root_dir = ""


class data_merge(object):

    def __init__(self, image_dir):
        self.dic = {}
        self.datasets = ["INSIGHTFACE", "PADDED", "JOINED"]
        self.image_dir = image_dir
        INSIGHTFACE = dataset_info()
        INSIGHTFACE.root_dir = os.path.join(self.image_dir, "sample_dataset")
        self.dic["INSIGHTFACE"] = INSIGHTFACE
        PADDED = dataset_info()
        PADDED.root_dir = os.path.join(self.image_dir, "sample_dataset")
        self.dic["PADDED"] = PADDED
        JOINED = dataset_info()
        JOINED.root_dir = os.path.join(self.image_dir, "sample_dataset")
        self.dic["JOINED"] = JOINED
        CUSTOM = dataset_info()
        CUSTOM.root_dir = os.path.join(self.image_dir, "sample_dataset")
        self.dic["Custom"] = CUSTOM



    def get_single_dataset(self, data_name="", train=True, img_size=256, map_size=32, transform=None, debug_subset_size=None, UUID=-1):
        if train:
            data_dir = self.dic[data_name].root_dir
            if data_name in self.datasets:
                data_set = Spoofing_custom(os.path.join(data_dir, "train_list.csv"), os.path.join(data_dir, "Train_files"), transform=transform, img_size=img_size, map_size=map_size, UUID=UUID)
            else:
                print("NO DATASET Found")
                exit()    
            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        else:
            data_dir = self.dic[data_name].root_dir
            if data_name in self.datasets:
                data_set = Spoofing_custom(os.path.join(data_dir, "val_list.csv"), os.path.join(data_dir, "Val_files"), transform=transform, img_size=img_size, map_size=map_size, UUID=UUID)
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
        else:
            print("No such protocol", protocol)
            exit()
        # elif protocol == "O_C_I_to_M":
        #     data_name_list_train = ["OULU", "CASIA_MFSD", "Replay_attack"]
        #     data_name_list_test = ["MSU_MFSD"]
        # elif protocol == "O_M_I_to_C":
        #     data_name_list_train = ["OULU", "MSU_MFSD", "Replay_attack"]
        #     data_name_list_test = ["CASIA_MFSD"]
        # elif protocol == "O_C_M_to_I":
        #     data_name_list_train = ["OULU", "CASIA_MFSD", "MSU_MFSD"]
        #     data_name_list_test = ["Replay_attack"]
        # elif protocol == "I_C_M_to_O":
        #     data_name_list_train = ["MSU_MFSD", "CASIA_MFSD", "Replay_attack"]
        #     data_name_list_test = ["OULU"] 
        # elif protocol == "M_I_to_C":
        #     data_name_list_train = ["MSU_MFSD", "Replay_attack"]
        #     data_name_list_test = ["CASIA_MFSD"]
        # elif protocol == "M_I_to_O":
        #     data_name_list_train = ["MSU_MFSD", "Replay_attack"]
        #     data_name_list_test = ["OULU"]
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
