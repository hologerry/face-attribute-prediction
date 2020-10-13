import os
from os.path import join as ospj

import torch
import torch.utils.data as data
from PIL import Image


class CelebAHQ(data.Dataset):
    def __init__(self, data_root, dataset_name, selected_attrs, transform_img, ):
        super().__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.selected_attrs = selected_attrs

        self.image_dir = ospj(self.data_root, dataset_name, 'CelebA-HQ-img')
        self.attr_file = ospj(self.data_root, dataset_name, 'CelebAMask-HQ-attribute-anno-skin.txt')

        self.transform_img = transform_img
        self.dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        self.num_image_attr_pairs = len(self.dataset)

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

        for i, line in enumerate(lines):
            split = line.strip().split()
            filename = split[0]
            values = split[1:]
            label = []
            for val in values:
                label.append(val == '1')
            self.dataset.append([filename, i])
        print(f'Finished preprocessing the {self.dataset_name} dataset...')

    def __getitem__(self, index):
        filename_a, index_a = self.dataset[index]
        image_a = Image.open(ospj(self.image_dir, filename_a))
        return self.transform_img(image_a), torch.LongTensor([index_a])

    def __len__(self):
        return self.num_image_attr_pairs
