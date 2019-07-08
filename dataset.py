# -*- coding: utf-8 -*-
# @Author   : Magic
# @Time     : 2019/7/5 10:23
# @File     : dataset.py

import torch
import glob
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from config import config_dict
from utils import map_label
from PIL import Image
import cv2

class SenseData(Dataset):
    def __init__(self, img_dir, transform = None):
        super(SenseData, self).__init__()
        self.img_list = glob.glob(img_dir+'/*/*.jpg')
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = self.img_list[index]
        _, label_map = map_label(config_dict['name_to_id'])
        img_label = label_map[img.split('/')[2]]
        img = Image.open(img)
        if self.transform:
            img = self.transform(img)

        return img, torch.Tensor([int(img_label)-1])


class SenseDataTest(Dataset):
    def __init__(self, img_dir, transform = None):
        super(SenseDataTest, self).__init__()
        self.img_list = sorted(glob.glob(img_dir+'/*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = self.img_list[index]
        filedir=img
        img = Image.open(img)
        if self.transform:
            img = self.transform(img)
        return index, filedir, img

if __name__ == '__main__':
    # train_data = SenseData(config_dict['data_dir_train'], None)
    # train_loader = DataLoader(train_data, batch_size=4, shuffle=False, num_workers=1)
    # for i, batch in enumerate(train_loader):
    #     print(batch[0], batch[1])
    test_data = SenseDataTest(config_dict['data_dir_test'], transforms.Compose([transforms.Resize(224, interpolation=2),transforms.ToTensor()]))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    for i, batch in enumerate(test_loader):
        print(batch.shape)
        break