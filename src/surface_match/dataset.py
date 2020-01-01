import json
import os
import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from albumentations import Compose, RandomBrightnessContrast
from tensorflow.python.keras import Model
from tensorflow.python.keras.preprocessing.image import save_img

from surface_match.config import FILE_NAME_VALID, FILE_NAME_TRAIN, SIZE_X, SIZE_Y, GROUP_COUNT, CURRENT_DIR


def column(matrix: list, i: int):
    return np.array([row[i] for row in matrix])


class BatchGenerator:
    def __init__(self):
        self.train: List = []
        self.train_weights: List[List[float]] = []
        self.train_weights_normalize: List[np.ndarray] = []

        self.valid: List = []
        self.valid_weights: List[List[float]] = []
        self.valid_weights_normalize: List[np.ndarray] = []

        self.hard_examples: List = []
        self.images: np.ndarray = np.array([])

        self.train_batch_size: int = 160
        self.valid_batch_size: int = 300
        self.samples_per_group: int = self.train_batch_size // GROUP_COUNT
        self.default_weight: float = 0.5

        self.train_weights_file = os.path.join(os.getcwd(), '..', '..', 'models', 'surface_match', 'train_weights.npz')

        self.load_dataset()
        self.init_weights()

    def load_dataset(self):
        (self.train, self.valid, self.images) = get_dataset(SIZE_X, SIZE_Y)

    def load_hard_examples(self):
        path = os.path.join(CURRENT_DIR, 'hard_indexes.json')

        with open(path, 'r') as read_file:
            hard_examples = json.load(read_file)

        self.hard_examples = [[] for _ in range(GROUP_COUNT)]
        for i in range(len(hard_examples)):
            hard_example = hard_examples[i]
            group = self.train[hard_example[0]]
            example = group[hard_example[1]]

            self.hard_examples[hard_example[0]].append(example)

    def get_batch(self, data_groups, weights):
        self.samples_per_group = self.train_batch_size // GROUP_COUNT

        images_1 = []
        images_2 = []
        results = []
        indexes = []

        for group_index in range(GROUP_COUNT):
            (group_images_1, group_images_2, group_results, group_indexes) =\
                self.get_group_examples(group_index, data_groups[group_index], weights[group_index])

            images_1.extend(group_images_1)
            images_2.extend(group_images_2)
            results.extend(group_results)
            indexes.extend(group_indexes)

        return images_1, images_2, results, indexes

    def get_batch_train(self):
        return self.get_batch(self.train, self.train_weights_normalize)

    def get_batch_valid(self):
        return self.get_batch(self.valid, self.valid_weights_normalize)

    def get_batch_hard(self):
        self.samples_per_group = self.train_batch_size // GROUP_COUNT

        data_groups = self.hard_examples[:]

        for group_index in range(GROUP_COUNT):
            group: List[int, int, float] = data_groups[group_index]
            examples_delta = self.samples_per_group - len(group)

            if examples_delta > 0:
                examples_from_train = random.choices(self.train[group_index], k=examples_delta)
                data_groups[group_index] = group + examples_from_train

        return self.get_batch(data_groups, self.train_weights)

    def get_group_examples(self, group_index, group, weights):
        if len(weights) == len(group):
            group_samples_indexes = np.random.choice(len(group), self.samples_per_group, p=weights)
        else:
            group_samples_indexes = np.random.choice(len(group), self.samples_per_group)

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

    def init_weights(self):
        # Train weights
        self.train_weights = []

        for train_group_index in range(len(self.train)):
            self.train_weights.append([])

            for i in range(len(self.train[train_group_index])):
                self.train_weights[train_group_index].append(self.default_weight)

        # Valid weights
        self.valid_weights = []

        for valid_group_index in range(len(self.valid)):
            self.valid_weights.append([])

            for i in range(len(self.valid[valid_group_index])):
                self.valid_weights[valid_group_index].append(self.default_weight)

    def init_weight_normalize(self):
        self.train_weights_normalize = []
        for weights in self.train_weights:
            weights_np = np.array(weights)
            weights_np /= weights_np.sum()
            self.train_weights_normalize.append(weights_np)

        self.valid_weights_normalize = []
        for weights in self.valid_weights:
            weights_np = np.array(weights)
            weights_np /= weights_np.sum()
            self.valid_weights_normalize.append(weights_np)

    def load_example_weights(self):
        if os.path.exists(self.train_weights_file):
            file_data = np.load(self.train_weights_file, allow_pickle=True)

            loaded_train_weights = file_data['data']
            loaded_train_weights_count = sum([len(listElem) for listElem in loaded_train_weights])

            inited_train_weights_count = sum([len(listElem) for listElem in self.train_weights])

            if loaded_train_weights_count == inited_train_weights_count:
                self.train_weights = loaded_train_weights

    def update_weights(self, samples: List[Tuple[int, int]], predicted: np.ndarray, real_results: List[float]):
        for i in range(len(samples)):
            predicted_value = predicted[i][0]
            error_delta = real_results[i] - predicted_value
            error_delta_sq = np.round(error_delta ** 2, 12)

            example_group = samples[i][0]
            example_index = samples[i][1]

            self.train_weights[example_group][example_index] = error_delta_sq

    def update_weights_by_model(self, model: Model, part=0.01):
        print('Start update_weights_by_model')
        train_examples_count = 0
        for i in range(len(self.train)):
            train_examples_count += len(self.train[i])

        batch_size_saved = self.train_batch_size
        self.train_batch_size = int(train_examples_count * part)

        (t_images_1, t_images_2, t_results, indexes) = self.get_batch_train()

        self.update_weights(indexes, model.predict(x=[t_images_1, t_images_2]), t_results)

        self.train_batch_size = batch_size_saved
        self.init_weight_normalize()
        print('Finish update_weights_by_model')

    def save_example_weights(self):
        print('Saving example weights')
        np.savez(self.train_weights_file, data=self.train_weights)
        print('Example weights are saved')


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
