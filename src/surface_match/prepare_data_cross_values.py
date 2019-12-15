import hashlib
import json
import os
import re

import numpy as np
from PIL import Image

######################################################################
#
# Script makes json-file with describe how many common surfaces
# on the second image from first and have structure like this:
# [
#   {
#     'scene': 1,
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
        'images_per_root': 800,
    },
}

DATA_SOURCES = [
    DATASET['archviz'],
]


def get_hash_image(image):
    string = 's' + str(image['scene']) + 'r' + str(image['root']) + 'f' + str(image['frame'])
    return hashlib.md5(string.encode('utf-8')).hexdigest()


def get_image_key(image):
    return str(image['scene']) + '.' + str(image['root'])


class DatasetCollector:
    def __init__(self, dataset):
        self.dataset = dataset

        # Path to json file with/for calculated data
        self.filepath = os.path.join(CURRENT_DIR, '..', '..', 'data', 'surface_match', dataset['code'] + '.json')

        # Exist already calculated data
        self.data = []

        # Hash list of exist data
        self.hash_list = []

        # List of completed roots
        self.completed_roots = []

        # List of exist scene folders
        self.scenes = []

        # List of exist image files
        self.image_files = []

        # Calculate values for image
        self.calc_initial_values()
        self.calc_values_for_images()

    def calc_initial_values(self):
        # Get exist file with calculated value from json file
        self.set_exist_file_data()

        # Collect folders with files
        self.set_scenes()

        # Collect image files
        self.set_image_files()

        # Define hash list for calculated images
        self.set_hash_list()

        # Define which root frames are fully calculated
        self.set_completed_roots()

    def calc_values_for_images(self):
        counter = 0

        for image_file in self.image_files:
            scene = image_file['scene']
            root = image_file['root']
            frame = image_file['frame']
            path = image_file['path']
            image_desc = 'Scene: ' + str(scene) + '. Root: ' + str(root) + '. Frame: ' + str(frame)

            if self.is_image_calculated(image_file):
                print('SKIP CALC FOR: ' + image_desc)
                continue

            print('DO CALC FOR: ' + image_desc)

            img = Image.open(path).convert('L')
            arr = np.array(img)
            flatten_arr = [x for sublist in arr for x in sublist]

            pixels = len(flatten_arr)
            white_sum = sum(flatten_arr)
            value = white_sum / 255 / pixels

            self.data.append({
                'scene': scene,
                'root': root,
                'frame': frame,
                'value': round(value, 4),
            })

            counter += 1

            if counter % 5000 == 0 and counter > 0:
                print('Save file partial with new records ' + str(counter))
                self.save_data()

        # Save data to json
        print('Save file complete with new records ' + str(counter))
        self.save_data()

    def set_exist_file_data(self):
        if os.path.exists(self.filepath):
            with open(self.filepath) as json_file:
                self.data = json.load(json_file)

    def set_scenes(self):
        self.scenes = []
        folders = os.listdir(self.dataset['images_dir'])

        for folder in folders:
            if os.path.isdir(os.path.join(self.dataset['images_dir'], folder)):
                self.scenes.append(folder)

    def set_image_files(self):
        self.image_files = []

        for scene in self.scenes:
            match_scene = re.findall(r'-(\d+)', scene)

            images = os.listdir(os.path.join(self.dataset['images_dir'], scene))

            for image in images:
                match = re.findall(r'root-(\d+)_frame-(\d+)', image)

                if match:
                    self.image_files.append({
                        'scene': int(match_scene[0]),
                        'path': os.path.join(self.dataset['images_dir'], scene, image),
                        'root': int(match[0][0]),
                        'frame': int(match[0][1]),
                    })

    def set_hash_list(self):
        self.hash_list = []

        for image in self.data:
            self.hash_list.append(get_hash_image(image))

    def set_completed_roots(self):
        self.completed_roots = []

        root_frames_counter = {}

        for item in self.data:
            key = get_image_key(item)

            try:
                root_frames_counter[key] += 1
            except KeyError:
                root_frames_counter[key] = 1

        for key in list(root_frames_counter.keys()):
            if root_frames_counter[key] == self.dataset['images_per_root']:
                self.completed_roots.append(key)

    def is_image_calculated(self, image):
        # Check whole calculated roots
        if get_image_key(image) in self.completed_roots:
            return True

        # Check image file personality
        if get_hash_image(image) in self.hash_list:
            return True

        return False

    def save_data(self):
        f = open(self.filepath, 'w', encoding='utf-8')
        json.dump(self.data, f, ensure_ascii=False, indent=2)
        f.close()


class DataCollector:
    def __init__(self, data_sources):
        for dataset in data_sources:
            DatasetCollector(dataset)


if __name__ == '__main__':
    DataCollector(DATA_SOURCES)
