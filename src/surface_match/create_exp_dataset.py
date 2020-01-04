import os

import numpy as np

from surface_match.config import FILE_NAME_VALID, FILE_NAME_TRAIN
from surface_match.dataset import BatchGenerator


def save(file_name: str, x1: list, x2: list, y: list, indexes: list):
    train_dir = os.path.join(os.getcwd(), '..', '..', 'train-data', 'surface_match')

    file = os.path.join(train_dir, file_name)
    np.savez(file, images_1=x1, images_2=x2, results=y, indexes=indexes)


batch_generator = BatchGenerator()
batch_generator.train_batch_size = 50
batch_generator.init_weights()
batch_generator.init_weight_normalize()

(images_1, images_2, results, indexes) = batch_generator.get_batch_train()
save(FILE_NAME_TRAIN, images_1, images_2, results, indexes)

(images_1, images_2, results, indexes) = batch_generator.get_batch_valid()
save(FILE_NAME_VALID, images_1, images_2, results, indexes)
