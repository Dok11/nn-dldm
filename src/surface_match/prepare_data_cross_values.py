import json
import os
import re

import numpy as np
from PIL import Image

######################################################################
#
# Script makes file with describe how many common surfaces
# on the second image from first and have structure like this:
# [
#   {
#     'root': '/path/to/image/root.png',
#     'frame': '/path/to/image/frame.png',
#     'value': 0
#   }, ...
# ]
#
# Where value is number in range [0..1]
#
######################################################################


CURRENT_DIR: str = os.getcwd()

DATASET = {
    'archviz': {
        'code': 'archviz',
        'images_dir': os.path.join(CURRENT_DIR, '..', '..', 'data', 'surface_match', 'archviz_images', 'cross'),
    },
}

DATA_SOURCES = [
    DATASET['archviz'],
]


class DataCollector:
    def __init__(self, data_sources):
        for dataset in DATA_SOURCES:
            filepath = os.path.join(CURRENT_DIR, '..', '..', 'data', 'surface_match', dataset['code'] + '.json')

            # Get exist file with calculated value from json file
            data = self.get_exist_file_data(filepath)

            # Collect folders with files
            scenes = self.get_scenes(dataset['images_dir'])

            # Collect image files
            image_files = self.get_image_files(scenes, dataset['images_dir'])

            # Calculate values for image
            counter = 0

            for image_file in image_files:
                exist_data_filter = lambda v: v['scene'] == image_file['scene']\
                                              and v['root'] == image_file['root']\
                                              and v['frame'] == image_file['frame']
                exist_data = next(filter(exist_data_filter, data), None)
                if exist_data:
                    continue

                print('Do calc for: ' + str(image_file))

                img = Image.open(image_file['path']).convert('L')
                arr = np.array(img)
                flatten_arr = [x for sublist in arr for x in sublist]

                pixels = len(flatten_arr)
                white_sum = sum(flatten_arr)
                value = white_sum / 255 / pixels

                data.append({
                    'scene': image_file['scene'],
                    'root': image_file['root'],
                    'frame': image_file['frame'],
                    'value': round(value, 4),
                })

                counter += 1

                if counter % 5000 == 0 and counter > 0:
                    print('Save file partial')
                    self.save_data(filepath, data)

            # Save data to json
            print('Save file complete')
            self.save_data(filepath, data)

    def get_exist_file_data(self, filepath):
        if os.path.exists(filepath):
            with open(filepath) as json_file:
                return json.load(json_file)

        return []

    def get_scenes(self, images_dir):
        scenes = []
        folders = os.listdir(images_dir)

        for folder in folders:
            if os.path.isdir(os.path.join(images_dir, folder)):
                scenes.append(folder)

        return scenes

    def get_image_files(self, scenes, images_dir):
        image_files = []

        for scene in scenes:
            match_scene = re.findall(r'-(\d+)', scene)

            images = os.listdir(os.path.join(images_dir, scene))

            for image in images:
                match = re.findall(r'root-(\d+)_frame-(\d+)', image)

                if match:
                    image_files.append({
                        'scene': int(match_scene[0]),
                        'path': os.path.join(images_dir, scene, image),
                        'root': int(match[0][0]),
                        'frame': int(match[0][1]),
                    })

        return image_files

    def save_data(self, filepath, data):
        f = open(filepath, 'w', encoding='utf-8')
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.close()


if __name__ == '__main__':
    DataCollector(DATA_SOURCES)
