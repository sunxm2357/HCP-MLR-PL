import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from PIL import Image
import xml.dom.minidom
from xml.dom.minidom import parse

import torch
import torch.utils.data as data


category_info = {"airplane": 0, "bare-soil": 1,  "buildings": 2, "cars": 3,	"chaparral": 4, "court": 5, "dock": 6,
                 "field": 7, "grass": 8, "mobile-home": 9, 	"pavement": 10, "sand": 11, "sea": 12, "ship": 13,
                 "tanks": 14, "trees": 15, "water": 16}


class BigEarth(data.Dataset):

    def __init__(self, mode,
                 img_dir, anno_path, labels_path,
                 input_transform=None, label_proportion=1.0):

        assert mode in ('train', 'val')

        self.mode = mode
        self.input_transform = input_transform
        self.label_proportion = label_proportion

        self.img_names = []
        self.img_dir = img_dir

        if mode == 'train':
            image_list_file = os.path.join(self.img_dir, 'multilabels', 'LandUse_Multilabeled_train.txt')
        else:
            image_list_file = os.path.join(self.img_dir, 'multilabels', 'LandUse_Multilabeled_test.txt')

        with open(image_list_file) as f:
            lines = f.readlines()
        lines = lines[1:]
        self.labels = []

        self.image_list = []
        for l in lines:
            tokens = l.split()
            self.image_list.append("%s/%s.tif" % (tokens[0][:-2], tokens[0]))
            label = tokens[1:]
            label = [int(a) for a in label]
            self.labels.append(label)
        self.labels = np.array(self.labels)

        # changedLabels : numpy.ndarray, shape->(len(self.img_names), 20)
        # value range->(-1 means label don't exist, 0 means not sure whether the label exists, 1 means label exist)
        self.changedLabels = self.labels
        if label_proportion != 1:
            print('Changing label proportion...')
            self.changedLabels = changeLabelProportion(self.labels, self.label_proportion)

    def __getitem__(self, index):
        input = Image.open( os.path.join(self.img_dir, 'UCMerced_LandUse', "Images", self.image_list[index])).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        return index, input, self.changedLabels[index], self.labels[index]

    def __len__(self):
        return len(self.image_list)


# =============================================================================
# Help Functions
# =============================================================================
def changeLabelProportion(labels, label_proportion):
    # Set Random Seed
    np.random.seed(0)

    mask = np.random.random(labels.shape)
    mask[mask < label_proportion] = 1
    mask[mask < 1] = 0
    label = mask * labels

    assert label.shape == labels.shape

    return label
