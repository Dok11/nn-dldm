import os

import numpy as np
import tensorflow as tf
from albumentations import Compose, RandomBrightnessContrast
from tensorflow.python.keras.preprocessing.image import save_img

from surface_match.config import FILE_NAME_VALID, FILE_NAME_TRAIN, SIZE_X, SIZE_Y, GROUP_COUNT


def column(matrix: list, i: int):
    return np.array([row[i] for row in matrix])


class BatchGenerator:
    def __init__(self):
        self.train: list = []
        self.valid: list = []
        self.images: np.ndarray = np.array([])
        self.train_batch_size = 120
        self.valid_batch_size = 300

        self.load_dataset()

    def load_dataset(self):
        (self.train, self.valid, self.images) = get_dataset(SIZE_X, SIZE_Y)

    def get_batch(self, train=True):
        samples_per_group = self.train_batch_size // GROUP_COUNT

        if train:
            data_groups = self.train
        else:
            data_groups = self.valid

        images_1 = []
        images_2 = []
        results = []
        indexes = []

        for group_index in range(GROUP_COUNT):
            group_samples_indexes = np.random.randint(0, len(data_groups[group_index]), samples_per_group)

            group_samples = []
            for i in range(len(group_samples_indexes)):
                group_sample_rnd_index = group_samples_indexes[i]
                group_samples.append(data_groups[group_index][group_sample_rnd_index])
                indexes.append([group_index, group_sample_rnd_index])

            group_images_1_idx = column(group_samples, 0)
            group_images_2_idx = column(group_samples, 1)
            group_images_1 = self.images[group_images_1_idx.astype(int)]
            group_images_2 = self.images[group_images_2_idx.astype(int)]
            group_results = column(group_samples, 2)

            images_1.extend(group_images_1)
            images_2.extend(group_images_2)
            results.extend(group_results)

        return images_1, images_2, results, indexes


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
