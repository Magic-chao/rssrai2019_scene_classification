# -*- coding: utf-8 -*-
# @Author   : Magic
# @Time     : 2019/7/4 12:01
# @File     : data.py

from config import config_dict
from dataset import SenseData, SenseDataTest
from torchvision import transforms

img_train = config_dict['data_dir_train']
img_val = config_dict['data_dir_val']
img_test = config_dict['data_dir_test']

def train_augs():
    return transforms.Compose([
        transforms.RandomResizedCrop(config_dict['im_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

def val_augs():
    return transforms.Compose([
        transforms.Resize(256, interpolation=2),
        transforms.CenterCrop(config_dict['im_size']),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

def test_augs():
    return transforms.Compose([
        transforms.Resize(256, interpolation=2),
        transforms.CenterCrop(config_dict['im_size']),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
def get_train_data(img_path=img_train, transform = train_augs()):
    return SenseData(img_path, transform)

def get_val_data(img_path=img_val, transform = val_augs()):
    return SenseData(img_path, transform)

def get_test_data(img_path=img_test, transform = test_augs()):
    return SenseDataTest(img_path, transform)

