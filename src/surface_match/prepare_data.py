import json
import os
import re

import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.utils import Progbar

from surface_match.config import GROUP_COUNT, SIZE_Y, SIZE_X, CHANNELS

# ============================================================================
# --- GLOBAL PARAMS ----------------------------------------------------------
# ----------------------------------------------------------------------------

CURRENT_DIR: str = os.getcwd()
DATASET = {
    'archviz': {
        'code': 'archviz',
        'images_real': os.path.join(CURRENT_DIR, '..', '..', 'data', 'surface_match', 'archviz_images', 'real'),
        'surface_match_file': os.path.join(CURRENT_DIR, '..', '..', 'data', 'surface_match', 'archviz.json'),
        'surface_match_data': [],
    },
    'simple': {
        'code': 'simple',
        'images_real': os.path.join(CURRENT_DIR, '..', '..', 'data', 'surface_match', 'simple_images', 'real'),
        'surface_match_file': os.path.join(CURRENT_DIR, '..', '..', 'data', 'surface_match', 'simple.json'),
        'surface_match_data': [],
    },
}
DATA_SOURCES = [
    DATASET['archviz'],
    DATASET['simple'],
]
VALIDATION_PART = 0.2


def get_image_as_np_array(path):
    if CHANNELS == 3:
        color_mode = 'rgb'
    else:
        color_mode = 'grayscale'

    img_loaded = load_img(path,
                          color_mode=color_mode,
                          target_size=(SIZE_Y, SIZE_X),
                          interpolation='bicubic')

    img = img_to_array(img_loaded)

    return np.array(img).astype('uint8')


class DataCollector:
    def __init__(self, data_sources):
        # PROPERTIES:
        # Information about sources for dataset
        self.data_sources = data_sources

        # List of path to real images
        self.images_dict = []
        self.images_dict_flip = []

        # List of real images in Numpy Array format (size_x, size_y, channels)
        self.images_np_arr = []

        # List of examples for dataset
        self.examples = []

        # List of grouped examples by 10 groups
        self.grouped_examples = []

        # ACTIONS:
        print('Load data from json file which generated from Blender file')
        self.load_json_data()

        print('Save all uses images as numpy array uint8')
        self.set_images()

        print('Set examples for dataset file')
        self.set_examples()

        print('Divide examples by distribution int the 10 groups')
        self.set_grouped_examples()

        print('Collect dataset and save into npz file')
        self.set_data()

    def load_json_data(self):
        for sources in self.data_sources:
            with open(sources['surface_match_file'], 'r') as read_file:
                sources['surface_match_data'] = json.load(read_file)

    def set_images(self):
        self.images_dict = []
        self.images_dict_flip = []
        self.images_np_arr = []

        # Collect image witch we need to use
        for source in self.data_sources:
            scene_dirs = os.listdir(source['images_real'])

            for folder in scene_dirs:
                files = os.listdir(os.path.join(source['images_real'], folder))

                for file in files:
                    image_path = os.path.join(source['images_real'], folder, file)
                    image_np_arr = get_image_as_np_array(image_path)

                    partial_path = re.sub(r'.+surface_match(.+)', r'\1', image_path)

                    self.images_dict.append(partial_path)
                    self.images_np_arr.append(image_np_arr)

        self.images_dict_flip = {self.images_dict[i]: i for i in range(0, len(self.images_dict))}

    def set_examples(self):
        # List of (index root image, index target image, cross value)
        # where index is index of image in the self.images_dict
        self.examples = []

        for source in self.data_sources:
            print('Start set_examples from ' + source['code'])

            progress_bar = Progbar(len(source['surface_match_data']))

            for item in source['surface_match_data']:
                root_frame_num = str(item['root']).zfill(4)
                image_root_path = '\\' + source['code'] + '_images\\real\\scene-' + str(item['scene']) + '\\' + root_frame_num + '.jpg'
                image_root_index = self.images_dict_flip[image_root_path]

                target_frame_num = str(item['frame']).zfill(4)
                image_target_path = '\\' + source['code'] + '_images\\real\\scene-' + str(item['scene']) + '\\' + target_frame_num + '.jpg'
                image_target_index = self.images_dict_flip[image_target_path]

                self.examples.append((image_root_index, image_target_index, item['value']))
                progress_bar.add(1)

    def set_grouped_examples(self):
        self.grouped_examples = [[] for i in range(GROUP_COUNT)]

        progress_bar = Progbar(len(self.examples))

        for example in self.examples:
            example_value = example[2]
            group = round(example_value * (GROUP_COUNT - 1))
            self.grouped_examples[group].append(example)
            progress_bar.add(1)

    def set_data(self):
        data_train = [[] for i in range(GROUP_COUNT)]
        data_valid = [[] for i in range(GROUP_COUNT)]

        for group_index in range(len(self.grouped_examples)):
            group = self.grouped_examples[group_index]

            for example_index in range(len(group)):
                example = group[example_index]

                if example_index % (1 / VALIDATION_PART):
                    data_train[group_index].append(example)
                else:
                    data_valid[group_index].append(example)

        train_dir = os.path.join(CURRENT_DIR, '..', '..', 'train-data', 'surface_match')

        if not (os.path.isdir(train_dir)):
            os.mkdir(train_dir)

        file = os.path.join(train_dir, 'data_' + str(SIZE_X) + 'x' + str(SIZE_Y))
        np.savez(file, train=data_train, valid=data_valid, images=self.images_np_arr)


if __name__ == '__main__':
    DataCollector(DATA_SOURCES)
