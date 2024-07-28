import cv2
import torch
import numpy as np
import torch.nn as nn
from libtiff import TIFF
from scipy import io
from torch.utils.data import DataLoader
from torchvision import transforms



Train_Rate = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
print('train rate is {}'.format(Train_Rate))

class DataPreprocess():
    def __init__(self,
                 ms4_url='./Data9/Image9/ms4.tif',
                 pan_url='./Data9/Image9/pan.tif',
                 label_np_url="./Data9/Image9/label.npy",
                 ms4_patch_size=16,
                 mode='r'
                 ):
        self.ms4_tif = TIFF.open(ms4_url, mode=mode)
        print('ms4 was successfully opened')
        self.pan_tif = TIFF.open(pan_url, mode=mode)
        print('pan was successfully opened')
        self.label_np = np.load(label_np_url)
        print('label_np was successfully opened')
        self.ms4_patch_size = ms4_patch_size

    def to_normalization(self, image):
        max_i = np.max(image)
        min_i = np.min(image)
        image = (image - min_i) / (max_i - min_i)
        return image

    def data_about(self):
        label_np = self.label_np - 1
        print(label_np.shape)
        # label_np = (self.label_np).astype(int) - 1
        label_element, element_count = np.unique(label_np, return_counts=True)
        # for i in range(len(label_element)):
            # print(label_element[i])
            # print(element_count[i])
        Categories_Number = len(label_element) - 1
        print('Categories Number: {}'.format(Categories_Number))
        label_row, label_column = np.shape(label_np)

        ground_xy = np.array([[]] * Categories_Number).tolist()
        ground_xy_all_data = np.arange(label_row * label_column * 2).reshape(label_row * label_column, 2)
        count = 0

        for row in range(label_row):
            for column in range(label_column):
                ground_xy_all_data[count] = [row, column]
                count = count + 1
                if label_np[row][column] != 255:
                    ground_xy[int(label_np[row][column])].append([row, column])

        for categories in range(Categories_Number):
            ground_xy[categories] = np.array(ground_xy[categories])
            shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
            np.random.shuffle(shuffle_array)

            ground_xy[categories] = ground_xy[categories][shuffle_array]
        shuffle_array = np.arange(0, label_row * label_column, 1)
        np.random.shuffle(shuffle_array)
        ground_xy_all_data = ground_xy_all_data[shuffle_array]

        ground_xy_train = []
        ground_xy_test = []
        label_train = []
        label_test = []

        for categories in range(Categories_Number):
            categories_number = len(ground_xy[categories])
            for i in range(categories_number):
                if i < int(categories_number * Train_Rate[categories]):
                    ground_xy_train.append(ground_xy[categories][i])
                else:
                    ground_xy_test.append(ground_xy[categories][i])
            label_train = label_train + [categories for x in range(int(categories_number * Train_Rate[categories]))]
            label_test = label_test + [categories for x in
                                       range(categories_number - int(categories_number * Train_Rate[categories]))]

        label_train = np.array(label_train)
        label_test = np.array(label_test)
        ground_xy_train = np.array(ground_xy_train)
        ground_xy_test = np.array(ground_xy_test)

        shuffle_array = np.arange(0, len(label_test), 1)
        np.random.shuffle(shuffle_array)
        label_test = label_test[shuffle_array]
        ground_xy_test = ground_xy_test[shuffle_array]

        shuffle_array = np.arange(0, len(label_train), 1)
        np.random.shuffle(shuffle_array)
        label_train = label_train[shuffle_array]
        ground_xy_train = ground_xy_train[shuffle_array]

        label_train = torch.from_numpy(label_train).type(torch.LongTensor)
        label_test = torch.from_numpy(label_test).type(torch.LongTensor)
        ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
        ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
        ground_xy_all_data = torch.from_numpy(ground_xy_all_data).type(torch.LongTensor)
        all_label = torch.cat((label_train, label_test), dim=0)
        all_ground_xy = torch.cat((ground_xy_train, ground_xy_test), dim=0)
        '''
        label_train: the label of train datasets
        label_test: the label of test datasets
        all_label: the label of all datasets, all label = label_train cats with label_test
        ground_xy_train: the coordinates of train datasets
        ground_xy_test: the coordinates of test datasets
        all_ground_xy: the coordinates of train datasets and test datasets 
        ground_xy_all_data: the coordinates of all datasets
        '''
        return label_train, label_test, all_label, ground_xy_train, ground_xy_test, all_ground_xy, ground_xy_all_data

    def get_ms_and_pan(self):
        Interpolation = cv2.BORDER_REFLECT_101
        ms4_np = self.ms4_tif.read_image()
        pan_np = self.pan_tif.read_image()
        ms4_patch_size = self.ms4_patch_size
        # Interpolation = cv2.BORDER_REFLECT_101
        top_size, bottom_size, left_size, right_size = (int(ms4_patch_size / 2 - 1), int(ms4_patch_size / 2),
                                                        int(ms4_patch_size / 2 - 1), int(ms4_patch_size / 2))
        ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)

        pan_patch_size = ms4_patch_size * 4
        top_size, bottom_size, left_size, right_size = (int(pan_patch_size / 2 - 4), int(pan_patch_size / 2),
                                                        int(pan_patch_size / 2 - 4), int(pan_patch_size / 2))
        pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)

        ms4 = self.to_normalization(ms4_np)
        pan = self.to_normalization(pan_np)
        pan = np.expand_dims(pan, axis=0)
        ms4 = np.array(ms4).transpose((2, 0, 1))

        # 将pan、ms4转换为torch.Tensor类型
        ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
        pan = torch.from_numpy(pan).type(torch.FloatTensor)

        return ms4, pan


# obj = DataPreprocess()
# label_train, label_test, all_label, ground_xy_train, ground_xy_test, all_ground_xy, ground_xy_allData = obj.data_about()
# ms4, pan = obj.get_ms_and_pan()

# print('the length of label_train: ', len(label_train))
# print('the size of label_train: ', label_train.size())
# print('the size of ground_xy_train: ', ground_xy_train.size())

# print('the length of label_test: ', len(label_test))
# print('the size of label_test: ', label_test.size())
# print('the size of ground_xy_test: ', ground_xy_test.size())

# print('the length of all_label: ', len(all_label))
# print('the size of all_label: ', all_label.size())
# print('the size of all_ground_xy: ', all_ground_xy.size())

"""
label_train      3130     torch.Size([3130])
ground_xy_train  3130     torch.Size([3130, 2])
label_test       310115   torch.Size([310115])
ground_xy_test   310115   torch.Size([310115, 2])
all_label        313245   torch.Size([313245])
all_ground_xy    313245   torch.Size([313245, 2])
ground_xy_allData         torch.Size([664000, 2])
"""

