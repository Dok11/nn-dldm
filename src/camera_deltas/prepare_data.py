import json
import os
import random

import numpy as np
from keras_preprocessing.image import load_img, img_to_array


RAD_TO_DEGREE = 57.2958  # 180/pi
SIZE_X = 90
SIZE_Y = 60
CURRENT_DIR: str = os.getcwd()
DATA_SLICE: int = 1
MAX_COUNT: int = 1000000  # per slice
NEIGHBOR_COUNT = 5
ANGLE_LIMIT = 60 / RAD_TO_DEGREE

DATASET = {
    'simple': {
        'images_dir': os.path.join(CURRENT_DIR, '..', '..', 'data', 'camera_deltas', 'simple_images', 'r'),
        'delta_path': os.path.join(CURRENT_DIR, '..', '..', 'data', 'camera_deltas', 'simple', 'data.json'),
        'files_data': [],
    },
    'classroom': {
        'images_dir': os.path.join(CURRENT_DIR, '..', '..', 'data', 'camera_deltas', 'classroom_images', 'r'),
        'delta_path': os.path.join(CURRENT_DIR, '..', '..', 'data', 'camera_deltas', 'classroom', 'data.json'),
        'files_data': [],
    },
}

DATA_SOURCES = [
    DATASET['simple'],
    # DATASET['classroom'],
]
VALIDATION_PART = 0.2


def get_image_as_np_array(path):
    image_loaded = load_img(path,
                            color_mode='grayscale',
                            target_size=(SIZE_Y, SIZE_X),
                            interpolation='bicubic')

    image = img_to_array(image_loaded)

    return np.array(image).astype('uint8')


class DataCollector:
    def __init__(self, data_sources):
        self.data_sources = data_sources
        self.images_dict = []
        self.images_np_arr = []

        # Load data from json file which generated from Blender file
        self.load_json_data()

        # Save all uses images as numpy array uint8
        self.set_images()

        # Collect dataset and save into npz file
        self.set_data()

    def load_json_data(self):
        for sources in self.data_sources:
            with open(sources['delta_path'], 'r') as read_file:
                sources['files_data'] = json.load(read_file)

    def set_images(self):
        for sources in self.data_sources:
            for file_data in sources['files_data']:
                image_path = os.path.join(sources['images_dir'], file_data['image_name'])
                image_np_arr = get_image_as_np_array(image_path)

                self.images_dict.append(image_path)
                self.images_np_arr.append(image_np_arr)

    def get_data_for_source(self, source):
        data = []  # [source image, destination, fov, result][]

        for file_data_index in range(len(source['files_data'])):
            source_image = source['files_data'][file_data_index]
            source_image_path = os.path.join(source['images_dir'], source_image['image_name'])
            source_image_index = self.images_dict.index(source_image_path)

            files_to_data = []

            # Find neighbors at left from source image in backward loop
            for i in range(NEIGHBOR_COUNT)[::-1]:
                destination = source['files_data'][file_data_index - i - 1]

                if destination and destination['scene'] == source_image['scene']:
                    files_to_data.append(destination)

            # Find neighbors at right from source image
            for i in range(NEIGHBOR_COUNT):
                try:
                    destination = source['files_data'][file_data_index + i + 1]

                    if destination and destination['scene'] == source_image['scene']:
                        files_to_data.append(destination)
                except IndexError:
                    print('out of range', IndexError)

            # Make delta data between source image and every destination
            for image_data in files_to_data:
                destination_image_path = os.path.join(source['images_dir'], image_data['image_name'])
                destination_image_index = self.images_dict.index(destination_image_path)

                rotation_delta = (
                    # source_image['rot_q']['w'] - image_data['rot_q']['w'],
                    # source_image['rot_q']['x'] - image_data['rot_q']['x'],
                    # source_image['rot_q']['y'] - image_data['rot_q']['y'],
                    # source_image['rot_q']['z'] - image_data['rot_q']['z'],

                    source_image['rot_e']['x'] - image_data['rot_e']['x'],
                    source_image['rot_e']['y'] - image_data['rot_e']['y'],
                    source_image['rot_e']['z'] - image_data['rot_e']['z'],
                )

                max_rotation_delta = max(rotation_delta)

                # Save delta data if rotation delta not too large
                if max_rotation_delta < ANGLE_LIMIT:
                    data.append((
                        source_image_index,  # source image index of self.images_np_arr
                        destination_image_index,  # and destination image index
                        source_image['fov'],
                        (  # full result
                            source_image['loc']['x'] - image_data['loc']['x'],
                            source_image['loc']['y'] - image_data['loc']['y'],
                            source_image['loc']['z'] - image_data['loc']['z'],
                        ) + rotation_delta,
                    ))

        return data

    def set_data(self):
        data_train = []  # [source image, destination, fov, result][]
        data_valid = []  # [source image, destination, fov, result][]

        for sources in self.data_sources:
            data = self.get_data_for_source(sources)

            for data_index in range(len(data)):
                if random.uniform(0, 1) <= VALIDATION_PART:
                    data_valid.append(data[data_index])
                else:
                    data_train.append(data[data_index])

        # todo: write working with slices
        slice_str = str(0).zfill(3)
        train_dir = os.path.join(CURRENT_DIR, '..', '..', 'train-data', 'camera_deltas')

        if not (os.path.isdir(train_dir)):
            os.mkdir(train_dir)

        file = os.path.join(train_dir, 'data_' + str(SIZE_X) + 'x' + str(SIZE_Y) + '_' + slice_str)
        np.savez(file, train=data_train, valid=data_valid, images=self.images_np_arr)


if __name__ == '__main__':
    DataCollector(DATA_SOURCES)
