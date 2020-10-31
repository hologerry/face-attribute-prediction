import random
from os.path import join as ospj

import torch
import torch.utils.data as data
from PIL import Image


def get_celebahq(celebahq_path, selected_attrs):
    print("Processing CelebAHQ dataset ...")
    celeba_attr_file = ospj(celebahq_path, 'CelebAMask-HQ-attribute-anno-skin.txt')
    img_path = ospj(celebahq_path, 'CelebA-HQ-img-256x256')
    with open(celeba_attr_file, 'r') as f:
        img_name_attrs_lines = f.readlines()

    attr2idx = {}
    idx2attr = {}

    all_attr_names = img_name_attrs_lines[1].split()
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    lines = img_name_attrs_lines[2:]
    random.seed(1234)
    random.shuffle(lines)

    celeba_test_dataset = []
    celeba_train_dataset = []

    for i, line in enumerate(lines):
        split = line.strip().split()
        filename = split[0]
        values = split[1:]
        label = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            label.append(values[idx] == '1')

        filepath = ospj(img_path, filename)
        pseudo = [1.0] * len(label)
        if i < 2000:
            celeba_test_dataset.append([filepath, label, pseudo, 1.0])
        else:  # 28000
            celeba_train_dataset.append([filepath, label, pseudo, 1.0])

    assert len(celeba_train_dataset) == 28000
    assert len(celeba_test_dataset) == 2000

    celeba_eval_part1 = celeba_test_dataset[:len(celeba_test_dataset)//2]
    celeba_eval_part2 = celeba_test_dataset[len(celeba_test_dataset)//2:]
    assert len(celeba_eval_part1) == 1000
    assert len(celeba_eval_part2) == 1000
    print("Processed CelebAHQ dataset")
    return celeba_train_dataset, celeba_test_dataset, celeba_eval_part1, celeba_eval_part2


def get_ffhq(ffhq_path, selected_attrs, prob_ratio):
    print("Processing FFHQ dataset ...")
    attr_file = ospj(ffhq_path, 'ffhq_attributes_list.txt')
    img_path = ospj(ffhq_path, 'images256x256')
    pseudo_label_path = ospj(ffhq_path, f"ffhq_pseudo_label_{prob_ratio}")
    with open(attr_file, 'r') as f:
        img_name_attrs_lines = f.readlines()

    attr2idx = {}
    idx2attr = {}

    all_attr_names = img_name_attrs_lines[1].split()
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    ffhq_test_dataset = []
    ffhq_train_dataset = []

    lines = img_name_attrs_lines[2:]
    for i, line in enumerate(lines):
        split = line.strip().split()
        filename = split[0]
        values = split[1:]

        pseudo_label_file = ospj(pseudo_label_path, filename.replace('.png', '.txt'))
        pseudo_all = []
        with open(pseudo_label_file) as f:
            lines = f.readlines()
            for line in lines:
                s = line.strip().split()
                pse = s[1]
                pseudo_all.append(float(pse))

        label = []
        pseudo = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            label.append(values[idx] == '1')
            pseudo.append(pseudo_all[idx])
        filepath = ospj(img_path, filename)

        if i >= 66000:
            ffhq_test_dataset.append([filepath, label, pseudo, 0.0])
        else:  # 4000
            ffhq_train_dataset.append([filepath, label, pseudo, 0.0])

    assert(len(ffhq_train_dataset)) == 66000
    assert(len(ffhq_test_dataset)) == 4000

    ffhq_eval_part1 = ffhq_test_dataset[:len(ffhq_test_dataset)//2]
    ffhq_eval_part2 = ffhq_test_dataset[len(ffhq_test_dataset)//2:]
    assert len(ffhq_eval_part1) == 2000
    assert len(ffhq_eval_part2) == 2000
    print("Processed FFHQ dataset")
    return ffhq_train_dataset, ffhq_test_dataset, ffhq_eval_part1, ffhq_eval_part2


def get_extra_celeba_images(extra_celeba_path, selected_attrs):
    print("Processing Extra CelebAHQ dataset ...")
    celeba_attr_file = ospj(extra_celeba_path, 'list_attr_skin_extra_imgs.txt')
    img_path = ospj(extra_celeba_path, 'img_celeba_256x256')
    with open(celeba_attr_file, 'r') as f:
        img_name_attrs_lines = f.readlines()

    attr2idx = {}
    idx2attr = {}

    all_attr_names = img_name_attrs_lines[1].split()
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    lines = img_name_attrs_lines[2:]
    celeba_test_dataset = []

    for i, line in enumerate(lines):
        if i >= 2000:
            break
        split = line.strip().split()
        filename = split[0]
        values = split[1:]
        label = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            label.append(values[idx] == '1')
        filepath = ospj(img_path, filename)
        pseudo = [1.0] * len(label)
        celeba_test_dataset.append([filepath, label, pseudo, 1.0])
    print("Processed Extra CelebA dataset")
    return celeba_test_dataset


def preprocess(celebahq_path, ffhq_path, extra_celeba_path, selected_attrs, ratio):
    celeba_train, celeba_test, celeba_evalp1, celeba_evalp2 = get_celebahq(celebahq_path, selected_attrs)
    ffhq_train, ffhq_test, ffhq_evalp1, ffhq_evalp2 = get_ffhq(ffhq_path, selected_attrs, ratio)
    extra_celeba_test = get_extra_celeba_images(extra_celeba_path, selected_attrs)
    train = celeba_train + ffhq_train
    assert len(train) == 28000 + 66000

    test = celeba_test + ffhq_test
    assert len(test) == 2000 + 4000

    attr_test = celeba_test + extra_celeba_test
    assert len(attr_test) == 2000 + 2000

    evalp1 = celeba_evalp1 + ffhq_evalp1
    assert len(evalp1) == 1000 + 2000

    evalp2 = celeba_evalp2 + ffhq_evalp2
    assert len(evalp2) == 1000 + 2000

    return train, test, attr_test, evalp1, evalp2


class CelebAHQFFHQPseudo(data.Dataset):
    def __init__(self, data_root, dataset_name, mode, transform_img, selected_attrs, ratio):
        super().__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.mode = mode
        self.transform_img = transform_img
        self.celebahq_ffhq = 'celebahq_ffhq' in self.dataset_name

        self.celebahq_path = ospj(data_root, 'celebahq')
        self.ffhq_path = ospj(data_root, 'ffhq')
        self.extra_celeba_path = ospj(data_root, 'celeba')

        self.train_list, self.test_list, self.attr_test_list, self.eval_p1_list, self.eval_p2_list = \
            preprocess(self.celebahq_path, self.ffhq_path, self.extra_celeba_path, selected_attrs, ratio)
        if self.mode == 'train':
            self.list = self.train_list
        elif self.mode == 'val' or self.mode == 'test':
            self.list = self.test_list
        elif self.mode == 'attr_test':
            self.list = self.attr_test_list
        elif self.mode == 'eval_part1':
            self.list = self.eval_p1_list
        elif self.mode == 'eval_part2':
            self.list = self.eval_p2_list
        else:
            raise ValueError
        self.length = len(self.list)

    def __getitem__(self, index):
        path_a, label_a, pseudo_a, is_celeba_a = self.list[index]
        image_a = Image.open(path_a)
        return {
            "img_a": self.transform_img(image_a), "attr_a": torch.FloatTensor(label_a),
            "pseudo_a": torch.FloatTensor(pseudo_a), "celeba_a": torch.FloatTensor([is_celeba_a]),
        }

    def __len__(self):
        return self.length
