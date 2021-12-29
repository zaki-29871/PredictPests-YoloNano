from torch.utils.data import Dataset, Subset
import torch
import tools
import os
from utils import *
import cv2
import xml.etree.ElementTree as ET
import utils
import numpy as np


class ClassTransformer:
    def __init__(self):
        name_list = self.get_name_list()
        self.name_to_id = {}
        self.id_to_name = {}

        for id, name in enumerate(name_list):
            self.name_to_id[name] = id
            self.id_to_name[id] = name

    def get_name_list(self):
        raise NotImplementedError()


class BugClassTransformer(ClassTransformer):
    def get_name_list(self):
        return ['tetrigoidea', 'fly']


class PaperClassTransformer(ClassTransformer):
    def get_name_list(self):
        return ['paper']


def rgb2bgr(x):
    y = np.zeros(x.shape, dtype=np.uint8)
    y[..., 0], y[..., 1], y[..., 2] = x[..., 2], x[..., 1], x[..., 0]
    return y


class DirtyBugs(Dataset):
    # Hint: the bugs is very dirty
    ROOT = 'D:\Dataset\YoloNano\data'

    def __init__(self, image_size=1024, ignore=True, gaussian_blur_kernel=None):
        self.size = len(os.listdir(os.path.join(self.ROOT, 'JPEGImages')))
        self.image_size = image_size
        self.size = 512
        # self.mask = np.ones(self.size, dtype=np.uint8)
        self.mask = np.zeros(self.size, dtype=np.uint8)
        self.gaussian_blur_kernel = gaussian_blur_kernel

        if ignore:
            # 20 沒tag蒼蠅
            self.mask[[1, 2, 4, 14, 15, 17, 19, 20, 23, 24, 25, 26]] = 1
            self.mask[[35, 36, 37, 38, 39, 40]] = 1
            self.mask[42:] = 1
            self.mask[[92, 93]] = 0

            # error = np.zeros(self.size, dtype=np.uint8)
            # error[self.mask == 0] = 1
            #
            # for x in error.nonzero():
            #     print(x + 1)

        self._make_mask_index()

    def __getitem__(self, index):
        index = self.mask_index[index] + 1
        X = cv2.imread(os.path.join(self.ROOT, 'JPEGImages', f'{index}.jpg'))
        X = rgb2bgr(X)

        if self.gaussian_blur_kernel is not None:
            X = cv2.GaussianBlur(X, (self.gaussian_blur_kernel, self.gaussian_blur_kernel), 0)

        targets = tools.load(os.path.join(self.ROOT, 'pkl/target', f'{index}.pkl'))

        X, targets, target_transformer = utils.letterbox_image(X, targets, self.image_size)

        X = torch.tensor(X)
        X = X.permute(2, 0, 1).float() / 255

        targets = torch.tensor(targets)
        index = torch.tensor(index).unsqueeze(0)

        return X, targets, index

    def __len__(self):
        return self.size

    def __str__(self):
        return type(self).__name__

    @staticmethod
    def get_class_transformer():
        return BugClassTransformer()

    @staticmethod
    def parse_annotation(root, index, class_transformer: ClassTransformer):
        filename = os.path.join(root, 'Annotations', f'{index}.xml')
        if not os.path.exists(filename):
            print(f'Cannot find {filename}')
            return

        root = ET.parse(filename).getroot()
        targets = []
        for object_tag in root.findall('./object'):
            class_id = class_transformer.name_to_id[object_tag.find('name').text]
            bbox = object_tag.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            targets.append([index, class_id, xmin, ymin, xmax - xmin, ymax - ymin])
        return targets

    @staticmethod
    def collate_fn(batch):
        images, targets, indexes = list(zip(*batch))
        images = torch.stack(images, dim=0)
        targets = torch.cat(targets, dim=0)
        indexes = torch.cat(indexes, dim=0)

        for i, index in enumerate(indexes):
            mask = targets[:, 0].int() == int(index)
            targets[mask, 0] = i

        return images, targets

    def _make_mask_index(self):
        self.size = np.sum(self.mask)
        self.mask_index = np.zeros(self.size, dtype=np.int)

        i = 0
        m = 0
        while i < len(self.mask):
            if self.mask[i]:
                self.mask_index[m] = i
                m += 1
            i += 1


class DirtyBugsUAV(Dataset):
    # Hint: the bugs is very dirty
    # ROOT = r'D:\Dataset\YoloNano\UAVimg\20190508'
    # ROOT = r'D:\Dataset\YoloNano\UAVimg\20190515B4'
    ROOT = r'D:\Dataset\YoloNano\UAVimg\20190515C1'
    # ROOT = r'D:\Dataset\YoloNano\UAVimg\20190522'

    def __init__(self, image_size=1024):
        self.filenames = os.listdir(self.ROOT)
        self.size = len(self.filenames)
        self.image_size = image_size

    def __getitem__(self, index):
        X = cv2.imread(os.path.join(self.ROOT, self.filenames[index]))
        X = rgb2bgr(X)

        X, target_transformer = utils.letterbox_image(X, target_size=self.image_size)

        X = torch.tensor(X)
        X = X.permute(2, 0, 1).float() / 255

        return X.unsqueeze(0)

    def __len__(self):
        return self.size

    def __str__(self):
        return type(self).__name__

    @staticmethod
    def get_class_transformer():
        return PaperClassTransformer()


class DirtyBugsPaper(Dataset):
    ROOT = r'D:\Dataset\YoloNano\BugPaper'

    def __init__(self, image_size=1024):
        self.size = 752
        self.image_size = image_size
        self.mask = np.zeros(self.size, dtype=np.uint8)

        for filename in os.listdir(os.path.join(self.ROOT, 'pkl/target')):
            index = int(filename.split('.')[0])
            self.mask[index] = 1
        self._make_mask_index()

    def __getitem__(self, index):
        index = self.mask_index[index]
        X = cv2.imread(os.path.join(self.ROOT, 'JPEGImages', f'{index}.png'))
        X = rgb2bgr(X)

        targets = tools.load(os.path.join(self.ROOT, 'pkl/target', f'{index}.pkl'))

        X, targets, target_transformer = utils.letterbox_image(X, targets, self.image_size)

        X = torch.tensor(X)
        X = X.permute(2, 0, 1).float() / 255

        targets = torch.tensor(targets)
        index = torch.tensor(index).unsqueeze(0)

        return X, targets, index

    def __len__(self):
        return self.size

    def __str__(self):
        return type(self).__name__

    def _make_mask_index(self):
        self.size = np.sum(self.mask)
        self.mask_index = np.zeros(self.size, dtype=np.int)

        i = 0
        m = 0
        while i < len(self.mask):
            if self.mask[i]:
                self.mask_index[m] = i
                m += 1
            i += 1

    @staticmethod
    def get_class_transformer():
        return PaperClassTransformer()

def clean_dirty_bugs_paper(dataset):
    size = len(dataset)
    valid = np.ones((size,), dtype=np.uint8)
    valid[[58, 59, 62, 63, 69, 70, 74, 75, 83, 84, 86, 111, 112, 120, 125, 126, 133, 134, 141, 143, 144, 168, 169, 170,
           171, 172, 173, 179, 192, 193, 215, 216, 217, 218, 219, 220, 221, 222, 223, 240, 248, 249, 362, 531, 532, 533,
           538, 539, 540, 543, 545]] = 0
    valid[546:] = 0
    return Subset(dataset, valid.nonzero()[0])

def random_subset(dataset, size, seed=None):
    assert size <= len(dataset), 'subset size cannot larger than dataset'
    np.random.seed(seed)
    indexes = np.arange(len(dataset))
    np.random.shuffle(indexes)
    indexes = indexes[:size]
    return Subset(dataset, indexes)


def random_split(dataset, train_ratio=0.8, seed=None):
    assert 0 <= train_ratio <= 1
    train_size = int(train_ratio * len(dataset))
    np.random.seed(seed)
    indexes = np.arange(len(dataset))
    np.random.shuffle(indexes)
    train_indexes = indexes[:train_size]
    test_indexes = indexes[train_size:]
    return Subset(dataset, train_indexes), Subset(dataset, test_indexes)
