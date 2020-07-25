import torch
import torch.utils.data as data

from PIL import Image
import os
import os.path


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CelebA(data.Dataset):
    def __init__(self, root, ann_file, transform=None, target_transform=None, loader=default_loader):
        images = []
        targets = []

        for line in open(os.path.join(root, ann_file), 'r'):
            sample = line.split()
            if len(sample) != 41:
                raise(RuntimeError("# Annotated face attributes of CelebA dataset should not be different from 40"))
            images.append(sample[0])
            targets.append([int(i) for i in sample[1:]])
        self.images = [os.path.join(root, 'img_align_celeba', img) for img in images]
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.images[index]
        sample = self.loader(path)
        target = self.targets[index]
        target = torch.LongTensor(target)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.images)


class CelebAEval(data.Dataset):
    def __init__(self, root, experiment_name, num_selected=13, transform=None, target_transform=None, loader=default_loader) -> None:
        # cureently we use 17 attributes, only the first 13 in the original 40
        # the last 4 is the skin color
        img_path = os.path.join(root, experiment_name, 'evaluate')
        label_path = os.path.join(root, experiment_name, 'evaluate_label')
        images = sorted(os.listdir(img_path))
        targets = sorted(os.listdir(label_path))
        assert len(images) == len(targets)
        self.images = [os.path.join(root, experiment_name, 'evaluate', img) for img in images]
        self.targets = [open(os.path.join(root, experiment_name, 'evaluate_label', label)).readlines()[:num_selected] for label in targets]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.images[index]
        sample = self.loader(path)
        target = self.targets[index]
        target = torch.LongTensor(target)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.images)
