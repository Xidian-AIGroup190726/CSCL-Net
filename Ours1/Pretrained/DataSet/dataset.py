import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyData(Dataset):
    def __init__(self, ms4, pan, label, xy, cut_size):
        self.train_data1 = ms4
        self.train_data2 = pan
        self.train_labels = label
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)
        y_pan = int(4 * y_ms)

        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        target = self.train_labels[index]
        return image_ms, image_pan, target, locate_xy, index

    def __len__(self):
        return len(self.gt_xy)


class MyData1(Dataset):
    def __init__(self, ms4, pan, xy, cut_size):
        self.train_data1 = ms4
        self.train_data2 = pan

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        return image_ms, image_pan, locate_xy, index

    def __len__(self):
        return len(self.gt_xy)


"""
xi'an datasets:

label_train      3130     torch.Size([3130])
ground_xy_train  3130     torch.Size([3130, 2])
label_test       310115   torch.Size([310115])
ground_xy_test   310115   torch.Size([310115, 2])
all_label        313245   torch.Size([313245])
all_ground_xy    313245   torch.Size([313245, 2])
"""

class GetDataLoader():
    def __init__(self, ms4=None, pan=None, patch_size=16, batch_size=64):
        self.ms4 = ms4
        self.pan = pan
        self.ms4PatchSize = patch_size
        self.batch_size = batch_size

    # train datasets: using MyData Class, return Loader and number of train datasets
    def get_train_data_loader(self, label_train=None, ground_xy_train=None):
        dataset = MyData(self.ms4, self.pan, label_train, ground_xy_train, self.ms4PatchSize)
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        n_data = len(dataset)
        print('number of train samples: {}'.format(n_data))  # 3130
        return data_loader, n_data

    # test datasets: using MyData Class, return Loader and number of test datasets
    def get_test_data_loader(self, label_test=None, ground_xy_test=None):
        dataset = MyData(self.ms4, self.pan, label_test, ground_xy_test, self.ms4PatchSize)
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        n_data = len(dataset)
        print('number of test samples: {}'.format(n_data))  # 310115
        return data_loader, n_data

    # all marked datasets: using MyData Class, return Loader and number of all marked datasets
    def get_all_mark_data_loader(self, all_label=None, all_ground_xy=None):
        dataset = MyData(self.ms4, self.pan, all_label, all_ground_xy, self.ms4PatchSize)
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        n_data = len(dataset)
        print('number of all mark samples: {}'.format(n_data))  # 313245
        return data_loader, n_data

    # all datasets: using MyData1 Class, return Loader and number of all datasets
    def get_all_data_loader(self, ground_xy_allData=None):

        # rate = 0.025
        # ground_xy_sample_data = []
        length = len(ground_xy_allData)

        # shuffle the ground_xy_allData
        shuffle_array = np.arange(0, length, 1)
        np.random.shuffle(shuffle_array)
        ground_xy_allData = ground_xy_allData[shuffle_array]

        # sample data into the array(ground_xy_sample_data)
        # for i in range(length):
        #     if i < int(length * rate):
        #         ground_xy_sample_data.append(ground_xy_allData[i])
        # print('the length of data for one stage is {}'.format(len(ground_xy_sample_data)))
        # dataset = MyData1(self.ms4, self.pan, ground_xy_sample_data, self.ms4PatchSize)
        dataset = MyData1(self.ms4, self.pan, ground_xy_allData, self.ms4PatchSize)
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        n_data = len(dataset)
        print('number of all samples: {}'.format(n_data))  # 664000
        return data_loader, n_data

