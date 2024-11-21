import os
import torch
from .Load_CSV_R import Spoofing_TrainVal, Spoofing_Test

class dataset_info(object):

    def __init__(self):
        self.root_dir = ""


class data_merge(object):

    def __init__(self):
        pass

# train_list - M1
# train_list1 - same as train_list but with different path to be convenient for docker 
# train_list2 - M1+M2
# train_list3 - M1+(M2w/o live)
# train_list4 - M1+M3 (all added only to train train)
# train_list5 - M1+M3
# train_list6 - M1+M2+M3
# train_list7 - M1+M3, min_shape >= 112
# train_list8 - M1+M4
# train_list9 - M1+M3+M4
# train_list10 - M1+M3+M4 - added more live, mainly D3 on which there were bad results (added to train only)
# train_list11 - same as train_list10 but with different path to be convenient for docker 
# train_list12 = 10% train_list11 
# train_list13 = M1+M3+M4+coid
# train_list14 = train_list13 + M4_synhtez
# train_list15 = train_list13 + 90%M4_synhtez, 10% added to val
# train_list16 = newly generated mask dataset


    def get_single_dataset(self, type='train', img_size=256, transform=None, debug_subset_size=None, UUID=-1):
        train_csv = os.path.join('./data', 'train_list16.csv')
        val_csv = os.path.join('./data', 'val_list16.csv')
        test_csv = os.path.join('./data', 'test_list16.csv')
        # test_csv = os.path.join('./data', 'test_list_mask.csv')

        if type == 'train':

            data_set = Spoofing_TrainVal(train_csv, transform=transform, img_size=img_size, UUID=UUID)

            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        elif type == 'val':

            data_set = Spoofing_TrainVal(val_csv, transform=transform, img_size=img_size, UUID=UUID)

            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        elif type == 'test':

            data_set = Spoofing_Test(test_csv, transform=transform, img_size=img_size, UUID=UUID)

            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        print("Loading dataset, number: {}".format(len(data_set)))
        return data_set

    def get_datasets(self, type='train', img_size=256, transform=None, debug_subset_size=None):
        
        data_set_sum = None
        if type == 'train':
            data_set_sum = self.get_single_dataset(type=type, img_size=img_size,  transform=transform, debug_subset_size=debug_subset_size, UUID=0)
        elif type == 'val':
            data_set_sum = self.get_single_dataset(type=type, img_size=img_size, transform=transform, debug_subset_size=debug_subset_size, UUID=0)
        elif type == 'test':
            data_set_sum = self.get_single_dataset(type=type, img_size=img_size, transform=transform, debug_subset_size=debug_subset_size, UUID=0)

        print("Total number: {}".format(len(data_set_sum)))
        return data_set_sum
