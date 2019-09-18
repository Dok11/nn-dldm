import json
import os
import re

import numpy as np
from keras_preprocessing.image import load_img, img_to_array

SIZE_X = 90
SIZE_Y = 60
CURRENT_DIR: str = os.getcwd()
DATA_SLICE: int = 1
MAX_COUNT: int = 1000000  # per slice

DATASET = {
    'simple': {
        'images_dir': os.path.join(CURRENT_DIR, '..', '..', 'data', 'camera_deltas', 'simple_images', 'r'),
        'delta_path': os.path.join(CURRENT_DIR, '..', '..', 'data', 'camera_deltas', 'simple', 'data.json'),
        'images_data': [],
        'delta_data': [],
    },
    'classroom': {
        'images_dir': os.path.join(CURRENT_DIR, '..', '..', 'data', 'camera_deltas', 'classroom_images', 'r'),
        'delta_path': os.path.join(CURRENT_DIR, '..', '..', 'data', 'camera_deltas', 'classroom', 'data.json'),
        'images_data': [],
        'delta_data': [],
    },
}

DATA_SOURCES = [
    # DATASET['simple'],
    DATASET['classroom'],
]
VALIDATION_PART = 0.2


class DataCollector:
    def __init__(self, data_sources):
        self.data_sources = data_sources

        self.set_images()
        self.set_deltas()
        self.set_data()

    def set_images(self):
        for sources in self.data_sources:
            file_list = os.listdir(sources['images_dir'])
            sources['images_data'] = []

            for file_name in file_list:
                name_parse = re.search(r'scene(\d+)_(\d+)\.', file_name)

                sources['images_data'].append({
                    'scene': int(name_parse.group(1)),
                    'frame': int(name_parse.group(2)),
                    'path': os.path.join(sources['images_dir'], file_name),
                })

    def get_image_as_np_array(self, path):
        image_loaded = load_img(path,
                                color_mode='grayscale',
                                target_size=(SIZE_Y, SIZE_X),
                                interpolation='bicubic')

        image = img_to_array(image_loaded)

        return np.array(image).astype('uint8')

    def set_deltas(self):
        for sources in self.data_sources:
            with open(sources['delta_path'], 'r') as read_file:
                sources['delta_data'] = json.load(read_file)

    def get_data_image_pairs(self, delta_item, images_data):
        image_pairs = []

        for image in images_data:
            image_target = 0
            image_previous = 0

            if image['frame'] == delta_item['frame']:
                image_target = image

            if image_target:
                image_previous = [x for x in images_data
                                  if x['frame'] == (delta_item['frame'] - 1) and x['scene'] == image_target['scene']]

                image_previous = image_previous[0]

            if image_target and image_previous:
                image_pairs.append((
                    self.get_image_as_np_array(image_target['path']),
                    self.get_image_as_np_array(image_previous['path']),
                    image_target['path'],
                    image_previous['path'],
                ))

        return image_pairs

    def get_data_for_source(self, source):
        data = []  # [current, previous, result][]

        for delta_item in source['delta_data']:
            image_pairs = self.get_data_image_pairs(delta_item, source['images_data'])

            for image_pair in image_pairs:
                data.append((
                    image_pair[0],  # current
                    image_pair[1],  # and previous frame
                    (
                        delta_item['loc']['x'],
                        delta_item['loc']['y'],
                        delta_item['loc']['z'],
                        delta_item['loc']['xy'],
                        delta_item['loc']['xz'],
                        delta_item['loc']['yx'],
                        delta_item['loc']['yz'],
                        delta_item['loc']['zx'],
                        delta_item['loc']['zy'],
                        delta_item['rot_q']['w'],
                        delta_item['rot_q']['x'],
                        delta_item['rot_q']['y'],
                        delta_item['rot_q']['z'],
                        delta_item['rot_e']['x'],
                        delta_item['rot_e']['y'],
                        delta_item['rot_e']['z'],
                    )
                ))

        return data

    def set_data(self):
        data_train = []  # [current, previous, result][]
        data_valid = []  # [current, previous, result][]

        for sources in self.data_sources:
            data = self.get_data_for_source(sources)

            for data_index in range(len(data)):
                if data_index % (1 / VALIDATION_PART) == 0:
                    data_valid.append(data[data_index])
                else:
                    data_train.append(data[data_index])

        # todo: write working with slices
        slice_str = str(0).zfill(3)
        train_dir = os.path.join(CURRENT_DIR, '..', '..', 'train-data', 'camera_deltas')

        if not (os.path.isdir(train_dir)):
            os.mkdir(train_dir)

        file = os.path.join(train_dir, 'data_' + str(SIZE_X) + 'x' + str(SIZE_Y) + '_' + slice_str)
        np.savez(file, train=data_train, valid=data_valid)


if __name__ == '__main__':
    DataCollector(DATA_SOURCES)
