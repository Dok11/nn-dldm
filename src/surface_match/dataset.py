import os

import numpy as np
from surface_match.config import FILE_NAME_VALID, FILE_NAME_TRAIN


def column(matrix: list, i: int):
    return np.array([row[i] for row in matrix])


def get_batch(data_groups: list, images: np.ndarray, train_batch_size: int, group_count: int):
    samples_per_group = train_batch_size // group_count

    images_1 = []
    images_2 = []
    results = []

    for group_index in range(group_count):
        group = np.array(data_groups[group_index])
        group_samples_indexes = np.random.randint(0, len(group), samples_per_group)
        group_samples = group[group_samples_indexes]

        group_images_1_idx = column(group_samples, 0)
        group_images_2_idx = column(group_samples, 1)
        group_images_1 = images[group_images_1_idx.astype(int)]
        group_images_2 = images[group_images_2_idx.astype(int)]
        group_results = column(group_samples, 2)

        for sample_index in range(len(group_results)):
            images_1.append(group_images_1[sample_index])
            images_2.append(group_images_2[sample_index])
            results.append(group_results[sample_index])

    return images_1, images_2, results


def get_experimental_dataset(use_train: bool):
    if use_train:
        file_name = FILE_NAME_TRAIN
    else:
        file_name = FILE_NAME_VALID

    file_path = os.path.join(os.getcwd(), '..', '..', 'train-data', 'surface_match', file_name + '.npz')
    file_data = np.load(file_path, allow_pickle=True)

    return file_data['images_1'], file_data['images_2'], file_data['results']


def get_dataset(x: int, y: int):
    file_name = 'data_' + str(x) + 'x' + str(y) + '.npz'
    file_path = os.path.join(os.getcwd(), '..', '..', 'train-data', 'surface_match', file_name)
    file_data = np.load(file_path, allow_pickle=True)

    return (
        file_data['train'],
        file_data['valid'],
        np.array(file_data['images']) / 255.,
    )
