import json
import os
import re
from random import randint
from typing import List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from surface_match.config import SIZE_X, SIZE_Y, GROUP_COUNT, CURRENT_DIR, FILE_NAME_DATA


def column(matrix: list, i: int):
    return np.array([row[i] for row in matrix])


class SurfaceMatchDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        self.data: List = []
        self.data_weights: List[List[float]] = []
        self.data_weights_normalize: List[np.ndarray] = []

        self.hard_examples: List = []
        self.use_hard_examples = False
        self.images: np.ndarray = np.array([])

        self.default_weight: float = 0.5
        self.data_dir = os.path.join(CURRENT_DIR, '..', '..', 'train-data', 'surface_match')

        self.train_weights_file = os.path.join(os.getcwd(), '..', '..', 'models', 'surface_match', 'train_weights.npz')

    def __len__(self):
        count = 0
        for i in range(len(self.data)):
            count += len(self.data[i])

        return count

    def __getitem__(self, idx):
        group_index = randint(0, GROUP_COUNT - 1)

        if self.use_hard_examples:
            group_weights = self.data_weights_normalize[group_index]
        else:
            group_weights = []

        (image_1, image_2, result, index) = self.get_sample(group_index, group_weights)

        sample = {
            'image_1': image_1,
            'image_2': image_2,
            'result': np.array(result, dtype=np.float32),
            'image_1_path': self.images[self.data[group_index][index][0]],
            'image_2_path': self.images[self.data[group_index][index][1]],
        }

        if self.transform:
            sample['image_1'] = self.transform(sample['image_1'])
            sample['image_2'] = self.transform(sample['image_2'])

        return sample

    def load_dataset(self):
        (self.data, self.images) = self.get_dataset(SIZE_X, SIZE_Y)

    def load_hard_examples(self):
        path = os.path.join(CURRENT_DIR, 'hard_indexes.json')

        with open(path, 'r') as read_file:
            hard_examples = json.load(read_file)

        self.hard_examples = [[] for _ in range(GROUP_COUNT)]
        for i in range(len(hard_examples)):
            hard_example = hard_examples[i]
            group = self.data[hard_example[0]]
            example = group[hard_example[1]]

            self.hard_examples[hard_example[0]].append(example)

    def get_sample(self, group_index, weights):
        group_len = len(self.data[group_index])

        if self.use_hard_examples and len(weights) == group_len:
            group_samples_index = np.random.choice(group_len, 1, p=weights)[0]
        else:
            group_samples_index = np.random.choice(group_len, 1)[0]

        img_idx_1, img_idx_2, result = self.data[group_index][group_samples_index]

        cur_dir = os.getcwd()

        # img_1_file_path = '\\'.join(list(map(str, filter(None, self.images[img_idx_1].split('\\')))))
        img_1_file_path = re.sub(r'^\\', r'', self.images[img_idx_1])
        # img_2_file_path = '\\'.join(list(map(str, filter(None, self.images[img_idx_2].split('\\')))))
        img_2_file_path = re.sub(r'^\\', r'', self.images[img_idx_2])

        img_1_path = os.path.join(cur_dir, '..', '..', 'data', 'surface_match', img_1_file_path)
        img_2_path = os.path.join(cur_dir, '..', '..', 'data', 'surface_match', img_2_file_path)

        group_image_1 = Image.open(img_1_path)
        group_image_2 = Image.open(img_2_path)

        return group_image_1, group_image_2, result, group_samples_index

    def init_weights(self):
        self.use_hard_examples = True

        # Weights
        self.data_weights = []

        for train_group_index in range(len(self.data)):
            self.data_weights.append([])

            for i in range(len(self.data[train_group_index])):
                self.data_weights[train_group_index].append(self.default_weight)

    def init_weight_normalize(self):
        self.data_weights_normalize = []
        for weights in self.data_weights:
            weights_np = np.array(weights)
            weights_np /= weights_np.sum()
            self.data_weights_normalize.append(weights_np)

    def load_example_weights(self):
        if os.path.exists(self.train_weights_file):
            file_data = np.load(self.train_weights_file, allow_pickle=True)

            loaded_train_weights = file_data['data']
            loaded_train_weights_count = sum([len(listElem) for listElem in loaded_train_weights])

            inited_train_weights_count = sum([len(listElem) for listElem in self.data_weights])

            if loaded_train_weights_count == inited_train_weights_count:
                self.data_weights = loaded_train_weights

    def get_dataset(self, x: int, y: int):
        file_path = os.path.join(self.data_dir, 'data_' + str(x) + 'x' + str(y) + '.npz')
        file_data = np.load(file_path, allow_pickle=True)

        return file_data['data'], file_data['images']

    def load_images(self, idx):
        loaded_images = []

        for image_num in idx:
            file_path = os.path.join(self.data_dir, 'images', str(image_num) + '.npz')
            file_data = np.load(file_path)
            loaded_images.append(file_data['arr_0'])

        return loaded_images

    def save_example_weights(self):
        print('Saving example weights')
        np.savez(self.train_weights_file, data=self.data_weights)
        print('Example weights are saved')


def get_experimental_dataset():
    file_path = os.path.join(os.getcwd(), '..', '..', 'train-data', 'surface_match', FILE_NAME_DATA + '.npz')
    file_data = np.load(file_path, allow_pickle=True)

    return file_data['images_1'], file_data['images_2'], file_data['results'], range(len(file_data['results']))
