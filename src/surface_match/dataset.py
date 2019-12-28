import json
import os
import random

import numpy as np
import tensorflow as tf
from albumentations import Compose, RandomBrightnessContrast
from tensorflow.python.keras.preprocessing.image import save_img

from surface_match.config import FILE_NAME_VALID, FILE_NAME_TRAIN, SIZE_X, SIZE_Y, GROUP_COUNT, CURRENT_DIR


def column(matrix: list, i: int):
    return np.array([row[i] for row in matrix])


class BatchGenerator:
    def __init__(self):
        self.train: list = []
        self.valid: list = []
        self.hard_examples: list = []
        self.images: np.ndarray = np.array([])
        self.train_batch_size = 160
        self.valid_batch_size = 300
        self.samples_per_group = self.train_batch_size // GROUP_COUNT

        self.load_dataset()

    def load_dataset(self):
        (self.train, self.valid, self.images) = get_dataset(SIZE_X, SIZE_Y)

    def load_hard_examples(self):
        path = os.path.join(CURRENT_DIR, 'hard_indexes.json')

        with open(path, 'r') as read_file:
            hard_examples = json.load(read_file)

        self.hard_examples = [[] for i in range(GROUP_COUNT)]
        for i in range(len(hard_examples)):
            hard_example = hard_examples[i]
            group = self.train[hard_example[0]]
            example = group[hard_example[1]]

            self.hard_examples[hard_example[0]].append(example)

    def get_batch(self, data_groups):
        self.samples_per_group = self.train_batch_size // GROUP_COUNT

        images_1 = []
        images_2 = []
        results = []
        indexes = []

        for group_index in range(GROUP_COUNT):
            (group_images_1, group_images_2, group_results, group_indexes) =\
                self.get_group_examples(group_index, data_groups[group_index])

            images_1.extend(group_images_1)
            images_2.extend(group_images_2)
            results.extend(group_results)
            indexes.extend(group_indexes)

        return images_1, images_2, results, indexes

    def get_batch_train(self):
        return self.get_batch(self.train)

    def get_batch_valid(self):
        return self.get_batch(self.valid)

    def get_batch_hard(self):
        self.samples_per_group = self.train_batch_size // GROUP_COUNT

        data_groups = self.hard_examples[:]

        for group_index in range(GROUP_COUNT):
            group: list[int, int, float] = data_groups[group_index]
            examples_delta = self.samples_per_group - len(group)

            if examples_delta > 0:
                examples_from_train = random.choices(self.train[group_index], k=examples_delta)
                data_groups[group_index] = group + examples_from_train

        return self.get_batch(data_groups)

    def get_group_examples(self, group_index, group):
        group_samples_indexes = np.random.randint(0, len(group), self.samples_per_group)

        group_indexes = []
        group_samples = []
        for i in range(len(group_samples_indexes)):
            group_sample_rnd_index = group_samples_indexes[i]
            group_samples.append(group[group_sample_rnd_index])
            group_indexes.append([group_index, group_sample_rnd_index])

        group_images_1_idx = column(group_samples, 0)
        group_images_2_idx = column(group_samples, 1)
        group_images_1 = self.images[group_images_1_idx.astype(int)]
        group_images_2 = self.images[group_images_2_idx.astype(int)]
        group_results = column(group_samples, 2)

        return group_images_1, group_images_2, group_results, group_indexes


def get_experimental_dataset(use_train: bool):
    if use_train:
        file_name = FILE_NAME_TRAIN
    else:
        file_name = FILE_NAME_VALID

    file_path = os.path.join(os.getcwd(), '..', '..', 'train-data', 'surface_match', file_name + '.npz')
    file_data = np.load(file_path, allow_pickle=True)

    return file_data['images_1'], file_data['images_2'], file_data['results']


def get_dataset(x: int, y: int):
    dir = os.path.dirname(os.path.abspath(__file__))
    file_name = 'data_' + str(x) + 'x' + str(y) + '.npz'
    file_path = os.path.join(dir, '..', '..', 'train-data', 'surface_match', file_name)
    file_data = np.load(file_path, allow_pickle=True)

    return (
        file_data['train'],
        file_data['valid'],
        np.array(file_data['images']) / 255.,
    )


def save_image(data, name):
    save_img(name, data)


def aug(p=0.5):
    return Compose([
        RandomBrightnessContrast(),
    ], p=p)


def loss_in_fact(y_true, y_pred):
    error = tf.math.subtract(y_pred, y_true)
    error_abs = tf.math.abs(error)
    return tf.math.reduce_mean(error_abs, axis=0)
