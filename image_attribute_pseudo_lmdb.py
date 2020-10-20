import os
from os.path import join as ospj

import lmdb
from PIL import Image
import torch
import torch.utils.data as data
import pyarrow as pa
import six


class ImageAttributePseudoLMDB(data.Dataset):
    def __init__(self, data_root, dataset_name, mode, transform_img):
        super().__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.mode = mode
        self.mode_dir = 'test' if mode == 'val' else mode   # 'val' is for testing pair
        self.transform_img = transform_img
        self.celebahq_ffhq = 'celebahq_ffhq' in self.dataset_name

        self.db_path = ospj(self.data_root, self.dataset_name, self.mode_dir)
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow_a = txn.get(self.keys[index])

        unpacked_a = pa.deserialize(byteflow_a)
        imgbuf_a = unpacked_a[0]
        buf_a = six.BytesIO()
        buf_a.write(imgbuf_a)
        buf_a.seek(0)
        image_a = Image.open(buf_a).convert('RGB')
        label_a = unpacked_a[1]
        pseudo_a = unpacked_a[2]
        is_celeba_a = unpacked_a[3]
        return {
            "img_a": self.transform_img(image_a), "attr_a": torch.FloatTensor(label_a),
            "pseudo_a": torch.FloatTensor(pseudo_a), "celeba_a": torch.FloatTensor([is_celeba_a]),
        }

    def __len__(self):
        return self.length
