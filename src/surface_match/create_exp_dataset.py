import os

import numpy as np

from surface_match.config import FILE_NAME_VALID, FILE_NAME_TRAIN, SIZE_X, SIZE_Y
from surface_match.dataset import get_batch, get_dataset


def save(file_name: str, x1: list, x2: list, y: list):
    train_dir = os.path.join(os.getcwd(), '..', '..', 'train-data', 'surface_match')

    file = os.path.join(train_dir, file_name)
    np.savez(file, images_1=x1, images_2=x2, results=y)


(train, valid, images) = get_dataset(SIZE_X, SIZE_Y)

batch_size = 30
group_count = 10

(images_1, images_2, results) = get_batch(train, images, batch_size, group_count)
save(FILE_NAME_TRAIN, images_1, images_2, results)

(images_1, images_2, results) = get_batch(valid, images, batch_size, group_count)
save(FILE_NAME_VALID, images_1, images_2, results)
