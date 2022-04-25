import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from libtiff import TIFF
import pandas as pd
import os
import os.path
import cv2
import torchvision


def make_dataset(image_list, labels, args):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        # if len(image_list[0].split()) > 2:
        #   images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        # else:
        if args == 'RDR':
            images = [(val.split()[0], 0 if int(val.split()[1]) < 2 else 1) for val in image_list]
        elif args == 'PDR':
            images = [(val.split()[0], 0 if int(val.split()[1]) < 4 else 1) for val in image_list]
        elif args == 'ABDR':
            images = [(val.split()[0], 0 if int(val.split()[1]) == 0 else 1) for val in image_list]
        elif args == 'C2':
            images = [(val.split()[0], 0 if int(val.split()[1]) >= 1 and int(val.split()[1]) <= 2 else 1) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

def v_loader(path):
    # with open(path, 'r') as f:
    # print(path)
    # with TIFF.open(path, 'r') as img:
    img = TIFF.open(path, 'r')
    a = np.array(list(img.iter_images()))
    sample = []
    for i in range(20):
        im = Image.fromarray(a[i]).convert('RGB')
        #     print(1)
        sample.append(np.array(im))
    return np.array(sample)


class ImageList(Dataset):
    def __init__(self, image_list, args, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels, args)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        elif mode == 'V':
            self.loader = v_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        # img = np.array(img)
        # print(img.shape)
        # img.reshape((224, 224, 3, 100))
        # print(img.shape)
        # PIL_image = Image.fromarray(img)
        # img1 = img
        if self.transform is not None:
            arr = np.array(1)
            try:
                img = self.transform(img)

            except:
                # a = type(img)
                # print(a)
                count = 0
                for i in range(len(img)):
                    count = count + 1
                    im = Image.fromarray(img[i])
                    im = self.transform(im)
                    # print(self.transform)
                    if count == 1:
                        total = im.unsqueeze(3)
                    else:
                                # print(count)
                        total = torch.cat((total, im.unsqueeze(3)), dim=-1)
                img = total
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


class ImageList_idx(Dataset):
    def __init__(self, image_list, args, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels, args)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)
