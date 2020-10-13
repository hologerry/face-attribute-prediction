import os
from os.path import join as ospj

import torch
import torch.utils.data as data
from PIL import Image


class Wild(data.Dataset):
    def __init__(self, data_root, dataset_name, mode, transform_img):
        super().__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.mode = mode

        self.image_dir = ospj(self.data_root, dataset_name, 'testset')
        self.transform_img = transform_img
        self.test_images = []

        self.preprocess()
        self.num_images = len(self.test_images)

    def preprocess(self):
        assert os.path.exists(self.image_dir), f'Image data directory does not exist: {self.image_dir}'
        self.test_images = sorted(os.listdir(self.image_dir))
        print(f'Finished preprocessing the {self.dataset_name} dataset...')

    def __getitem__(self, index):
        if self.mode == 'test':
            filename = self.test_images[index]
            image = Image.open(ospj(self.image_dir, filename))
            image = image.convert('RGB')
        else:
            image = None
            raise NotImplementedError
        return self.transform_img(image), torch.LongTensor([index])

    def __len__(self):
        return self.num_images
