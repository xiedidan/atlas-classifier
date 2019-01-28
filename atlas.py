import sys
import os
import torch
import numpy as np
import random
import csv
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image

TRAIN_LIST_NAME = 'train.csv'
IMG_COLORS = [
    'green', # protein of interest
    'blue', # nucleus
    'red', # microtubules
    'yellow', # endoplasmic reticulum
]

def collater(data):
    imgs = [s['image'] for s in data]
    annos = [s['anno'] for s in data]

    imgs = torch.stack(imgs)
    annos = torch.stack(annos)

    return {
        'images': imgs,
        'annos': annos
    }

class CsvDataset(Dataset):
    def __init__(self, csv_path, data_root, num_classes, phase='train', label='multi', augment=None):
        self.csv_path = csv_path
        self.data_root = data_root
        self.augment = augment
        self.phase = phase
        self.label = label
        self.num_classes = num_classes

        if self.phase == 'train' or self.phase == 'val':
            self.anno_df = pd.read_csv(os.path.join(self.data_root, TRAIN_LIST_NAME))
            self.image_path = os.path.join(self.data_root, 'train')
        else: # test
            self.image_path = os.path.join(self.data_root, 'test')
        
        self.image_ids = list(pd.read_csv(self.csv_path)['id'])

        self.samples = []
        for image_id in self.image_ids:
            sample_files = []
            self.samples.append(sample_files)
        
        annos = list(self.anno_df['Target'])
        self.annotations = [[int(pos) for pos in anno.split(' ')] for anno in annos]

    def _load_image(self, index):
        img_id = self.image_ids[index]
        image_files = [os.path.join(self.image_path, '{}_{}.png'.format(img_id, color)) for color in IMG_COLORS]

        layers = [skimage.io.imread(img_file) for img_file in image_files]

        # image.shape = [M, N, 4]
        image = np.stack(layers, axis=2)
        image = transforms.functional.to_pil_image(image)

        return image

    def _load_multi_anno(self, index):
        img_id = self.image_ids[index]
        labels = self.anno_df[self.anno_df['Id'] == img_id].iloc[0]['Target'].split(' ')
        labels = [int(label) for label in labels]

        targets = [1 if j in labels else 0 for j in range(self.num_classes)]

        return np.array(targets)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image = self._load_image(index)
        anno = torch.from_numpy(self._load_multi_anno(index))

        if self.augment is not None:
            image = self.augment(image)

        return {
            'image': image,
            'anno': anno
        }
