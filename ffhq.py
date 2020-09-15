import os
from os.path import join as ospj

import torch
import torch.utils.data as data
from PIL import Image


class FFHQ(data.Dataset):
    def __init__(self, data_root, dataset_name, transform_img):
        super().__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name

        self.image_dir = ospj(self.data_root, dataset_name, 'images1024x1024')

        self.transform_img = transform_img
        self.images = []
        self.image_idxes = []

        self.preprocess()
        self.num_images = len(self.images)

    def preprocess(self):
        assert os.path.exists(self.image_dir), f'Image data directory does not exist: {self.image_dir}'
        for idx in range(70000):
            img_sub_dir = f'{(idx // 1000):02d}000'
            img_file_name = f'{idx:05d}.png'
            img_path = ospj(img_sub_dir, img_file_name)
            self.images.append(img_path)
            self.image_idxes.append(idx)
        print(f'Finished preprocessing the {self.dataset_name} dataset...')

    def __getitem__(self, index):
        filename = self.images[index]
        file_idx = self.image_idxes[index]
        image = Image.open(ospj(self.image_dir, filename))
        return self.transform_img(image), torch.LongTensor([file_idx])

    def __len__(self):
        return self.num_images
