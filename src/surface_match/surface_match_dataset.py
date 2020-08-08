import json
import os
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
        self.data_valid: List = []
        self.data_weights: List[List[float]] = []
        self.data_weights_normalize: List[np.ndarray] = []

        self.hard_examples: List = []
        self.use_hard_examples = False
        self.use_valid = False
        self.images: List = []
        self.images_path: List = []

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

        (img_1, img_2, result, index) = self.get_sample(group_index, group_weights)

        img_idx_1 = self.data[group_index][index][0]
        img_idx_2 = self.data[group_index][index][1]

        sample = {
            'image_1': Image.fromarray(img_1),
            'image_2': Image.fromarray(img_2),
            'result': np.array(result, dtype=np.float32),
        }

        if self.transform:
            sample['image_1'] = self.transform(sample['image_1'])
            sample['image_2'] = self.transform(sample['image_2'])

        return sample

    def load_dataset(self):
        (self.data, self.data_valid, self.images, self.images_path)\
            = self.get_dataset(SIZE_X, SIZE_Y)

        # for i in range(len(images)):
        #     self.images.append(Image.fromarray(images[i]))

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
        if self.use_valid:
            group = self.data_valid[group_index]
        else:
            group = self.data[group_index]

        group_len = len(group)

        if self.use_hard_examples and len(weights) == group_len:
            sample_index = np.random.choice(group_len, 1, p=weights)[0]
        else:
            sample_index = np.random.choice(group_len, 1)[0]

        img_idx_1, img_idx_2, result = group[sample_index]

        img_1 = self.images[img_idx_1]
        img_2 = self.images[img_idx_2]

        return img_1, img_2, result, sample_index

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

        return file_data['data'], file_data['valid'], file_data['images'], file_data['images_path']

    def save_example_weights(self):
        print('Saving example weights')
        np.savez(self.train_weights_file, data=self.data_weights)
        print('Example weights are saved')


def get_experimental_dataset():
    file_path = os.path.join(os.getcwd(), '..', '..', 'train-data', 'surface_match', FILE_NAME_DATA + '.npz')
    file_data = np.load(file_path, allow_pickle=True)

    return file_data['images_1'], file_data['images_2'], file_data['results'], range(len(file_data['results']))
