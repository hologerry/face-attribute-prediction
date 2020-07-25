import os
import random
from os.path import join as ospj

import torch
import torch.utils.data as data
from PIL import Image


class CelebAMaskHQ(data.Dataset):
    def __init__(self, data_root, dataset_name, mode, transform_img, transform_seg=None, selected_attrs=None, seg_channel=0):
        super().__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.mode = mode
        self.selected_attrs = selected_attrs
        # we use all 44 attribute to train the classifier
        self.seg_channel = seg_channel

        self.image_dir = ospj(self.data_root, dataset_name, 'CelebA-HQ-img')
        self.attr_file = ospj(self.data_root, dataset_name, 'CelebAMask-HQ-attribute-anno-skin.txt')

        self.transform_img = transform_img
        self.transform_seg = transform_seg
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        self.num_image_attr_pairs = len(self.train_dataset) if self.mode == 'train' else len(self.test_dataset)

    def preprocess(self):
        assert os.path.exists(self.image_dir), f'Image data directory does not exist: {self.image_dir}'
        assert os.path.exists(self.attr_file), f'Attribute file does not exist: {self.attr_file}'
        with open(self.attr_file, 'r') as f:
            img_name_attrs_lines = f.readlines()
        all_attr_names = img_name_attrs_lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        lines = img_name_attrs_lines[2:]

        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.strip().split()
            filename = split[0]
            values = split[1:]
            label = []
            # for attr_name in self.selected_attrs:
            #     idx = self.attr2idx[attr_name]
            #     label.append(values[idx] == '1')
            for val in values:
                label.append(val == '1')
            if i < 2000:
                self.test_dataset.append([filename, label])
            else:  # 28000
                self.train_dataset.append([filename, label])
        print(f'Finished preprocessing the {self.dataset_name} dataset...')

    def __getitem__(self, index):
        if self.mode == 'train':
            filename_a, label_a = self.train_dataset[index]
            image_a = Image.open(ospj(self.image_dir, filename_a))
            return self.transform_img(image_a), torch.LongTensor(label_a)

        elif self.mode == 'val':
            filename_a, label_a = self.test_dataset[index]
            image_a = Image.open(ospj(self.image_dir, filename_a))
            return self.transform_img(image_a), torch.LongTensor(label_a)

        else:
            filename_a, label_a = self.test_dataset[index]
            image_a = Image.open(ospj(self.image_dir, filename_a))
            return self.transform_img(image_a), torch.LongTensor(label_a)

    def __len__(self):
        return self.num_image_attr_pairs
