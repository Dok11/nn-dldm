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
TRAIN_DATA_PART = 15


class DataCollector:
    def __init__(self):
        self.images = self.get_images()
        self.deltas = self.get_deltas()

        self.set_data()

    def get_images(self):
        result = []

        dir_root = os.path.join(CURRENT_DIR, '..', '..', 'data', 'simple_images', 'r', 'JPEG')
        file_list = os.listdir(dir_root)

        for file_name in file_list:
            name_parse = re.search(r'scene(\d+)_(\d+)\.', file_name)

            result.append({
                'scene': int(name_parse.group(1)),
                'frame': int(name_parse.group(2)),
                'path': os.path.join(dir_root, file_name),
            })

        return result

    def get_image_as_np_array(self, path):
        image_loaded = load_img(path,
                                color_mode='grayscale',
                                target_size=(SIZE_Y, SIZE_X),
                                interpolation='bicubic')

        image = img_to_array(image_loaded)

        return np.array(image).astype('uint8')

    def get_deltas(self):
        file_path = os.path.join(CURRENT_DIR, '..', '..', 'data', 'simple', 'data.json')

        with open(file_path, 'r') as read_file:
            result = json.load(read_file)

        return result

    def get_data_image_pairs(self, delta_item):
        image_pairs = []

        for image in self.images:
            image_target = 0
            image_previous = 0

            if image['frame'] == delta_item['frame']:
                image_target = image

            if image_target:
                image_previous = [x for x in self.images
                                  if x['frame'] == (delta_item['frame'] - 1) and x['scene'] == image_target['scene']]

                image_previous = image_previous[0]

            if image_target and image_previous:
                image_pairs.append((
                    # image_target['path'],
                    # image_previous['path'],
                    self.get_image_as_np_array(image_target['path']),
                    self.get_image_as_np_array(image_previous['path']),
                ))

        return image_pairs

    def set_data(self):
        for slice_index in range(DATA_SLICE):
            print('start work slice with', slice_index, 'of', DATA_SLICE)
            inputs_x1 = []  # current
            inputs_x2 = []  # previous
            outputs = []

            count = 0

            for delta_item in self.deltas[slice_index::DATA_SLICE]:
                image_pairs = self.get_data_image_pairs(delta_item)

                for image_pair in image_pairs:
                    inputs_x1.append(image_pair[0])  # current
                    inputs_x2.append(image_pair[1])  # and previous frame
                    outputs.append((
                        # delta_item['loc']['x'],
                        # delta_item['loc']['y'],
                        # delta_item['loc']['z'],
                        # delta_item['rot']['w'],
                        delta_item['rot']['x'],
                        delta_item['rot']['y'],
                        delta_item['rot']['z'],
                    ))

                    # inverted
                    inputs_x1.append(image_pair[1])  # current
                    inputs_x2.append(image_pair[0])  # and previous frame
                    outputs.append((
                        # -delta_item['loc']['x'],
                        # -delta_item['loc']['y'],
                        # -delta_item['loc']['z'],
                        -delta_item['rot']['x'],
                        -delta_item['rot']['y'],
                        -delta_item['rot']['z'],
                    ))

                count += 1
                if count >= MAX_COUNT:
                    break

            slice_str = str(slice_index).zfill(3)
            train_dir = os.path.join(CURRENT_DIR, '..', '..', 'train-data', 'deltas')

            if not (os.path.isdir(train_dir)):
                os.mkdir(train_dir)

            file = os.path.join(train_dir, 'data_' + slice_str)
            np.savez(file, x1=inputs_x1, x2=inputs_x2, y=outputs)


if __name__ == '__main__':
    DataCollector()
