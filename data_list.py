#from __future__ import print_function, division

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
import torch.utils.data as data
import os
import os.path
import kornia as K


def make_dataset(image_list, labels):
    if labels:  # labels=None for imagenet
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:      # split and get the labels
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        # images = [(val.split()[0], int(val.split()[1])) for val in image_list]
        images = [val.strip() for val in image_list]
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    #from torchvision import get_image_backend
    #if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    #else:
        return pil_loader(path)


class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_data, preprocess=None, with_freq=False, labels=None, transform=None, target_transform=None,
                 loader=default_loader, ret_path=False):  # ImageList(image_list = '../data/imagenet/train.txt')
        if not with_freq:
            imgs = make_dataset(image_data, labels)
        else:
            imgs = image_data[:, 0].tolist()
            self.freq  = image_data[:, 1].tolist()
        self.with_freq = with_freq
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.ret_path = ret_path
        self.preprocess = preprocess

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # path, target = self.imgs[index]
        img = self.loader(self.imgs[index])
        # img = K.io.load_image(self.imgs[index], K.io.ImageLoadType.RGB32, device="cpu")


        if self.preprocess is not None:
            img = self.preprocess(img)

        elif self.transform is not None:
            img = self.transform(img)


        if self.ret_path:
            return img, self.imgs[index]
        elif self.with_freq:
            return img, int(self.freq[index])
        else:
            return img

    def __len__(self):
        return len(self.imgs)

