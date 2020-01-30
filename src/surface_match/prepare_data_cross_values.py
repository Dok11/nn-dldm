import json
import os
import re

import numpy as np
from PIL import Image
from tensorflow.python.keras.utils import Progbar

######################################################################
#
# Script makes json-files (one per scene) with describe
# how many common surfaces on the second image from first
# and have structure like this:
# [
#   [
#     0: 1,
#     1: '/path/to/image/root.png',
#     2: '/path/to/image/frame.png',
#     3: 0
#   ], ...
# ]
#
# Where:
# [0] - Scene
# [1] - Root
# [2] - Frame
# [3] - Value. is number in range [0..1]
#
######################################################################

CURRENT_DIR: str = os.getcwd()

DATASET = {
    'archviz': {
        'code': 'archviz',
        'images_dir': os.path.join(CURRENT_DIR, '..', '..', 'data', 'surface_match', 'archviz_images', 'cross'),
        'images_per_root': 800,
    },
    'classroom': {
        'code': 'classroom',
        'images_dir': os.path.join(CURRENT_DIR, '..', '..', 'data', 'surface_match', 'classroom_images', 'cross'),
        'images_per_root': 800,
    },
    'simple': {
        'code': 'simple',
        'images_dir': os.path.join(CURRENT_DIR, '..', '..', 'data', 'surface_match', 'simple_images', 'cross'),
        'images_per_root': 500,
    },
}

DATA_SOURCES = [
    DATASET['archviz'],
    DATASET['classroom'],
    DATASET['simple'],
]


def get_image_key_for_scene(image):
    return str(image[0])


def get_image_key_for_root(image):
    # Scene and root
    return str(image[0]) + '.' + str(image[1])


def get_image_key_for_image(image):
    # Scene, root and frame
    return 's' + str(image[0]) + 'r' + str(image[1]) + 'f' + str(image[2])


class DatasetCollector:
    def __init__(self, dataset):
        self.dataset = dataset

        # Path to json file with/for calculated data
        self.file_path = os.path.join(CURRENT_DIR, '..', '..', 'data', 'surface_match', dataset['code'] + '.json')

        # Exist already calculated data
        self.data = []

        # List of completed scenes
        self.completed_scenes = []

        # List of completed roots
        self.completed_roots = []

        # List of completed images
        self.completed_images = []

        # List of exist scene folders
        self.scenes = []

        # List of exist image files
        self.image_files = []

        # ACTIONS:
        self.calc_initial_values()
        self.calc_values_for_images()

    def calc_initial_values(self):
        # Get exist file with calculated value from json file
        self.set_exist_file_data()

        # Collect folders with files
        self.set_scenes()

        # Collect image files
        self.set_image_files()

        # Define which scenes, root frames and images are fully calculated
        self.set_completed_scenes()
        self.set_completed_roots()
        self.set_completed_images()

    def calc_values_for_images(self):
        print('Run calc_values_for_images()')
        counter = 0

        # Progress bar for remain items to calculate
        progress_bar = Progbar(len(self.image_files) - len(self.data))

        for image_file in self.image_files:
            scene = image_file['scene']
            root = image_file['root']
            frame = image_file['frame']
            path = image_file['path']
            image_file_data = (scene, root, frame)

            if self.is_image_calculated(image_file_data):
                continue

            img = Image.open(path)
            img.thumbnail((16, 16))
            value = np.array(img).mean() / 255

            self.data.append((
                scene,
                root,
                frame,
                round(value, 4),
            ))

            counter += 1

            if counter % 500 == 0:
                progress_bar.add(500)

            if counter % 25000 == 0 and counter > 0:
                print('\nSave file partial with new records ' + str(counter))
                self.save_data()

        # Save data to json
        print('\nSave file complete with new records ' + str(counter))
        self.save_data()

    def set_exist_file_data(self):
        print('Run set_exist_file_data()')
        if os.path.exists(self.file_path):
            with open(self.file_path) as json_file:
                self.data = json.load(json_file)

        print('Exist data about ' + str(len(self.data)) + ' files\n')

    def set_scenes(self):
        print('Run set_scenes()')
        self.scenes = []
        folders = os.listdir(self.dataset['images_dir'])

        for folder in folders:
            if os.path.isdir(os.path.join(self.dataset['images_dir'], folder)):
                self.scenes.append(folder)

        print('Define ' + str(len(self.scenes)) + ' scenes\n')

    def set_image_files(self):
        print('Run set_image_files()')
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

        print('Define data about ' + str(len(self.image_files)) + ' images\n')

    def set_completed_scenes(self):
        print('Run set_completed_scenes()')
        self.completed_scenes = []

        scene_frames_counter = {}

        for item in self.data:
            key = get_image_key_for_scene(item)

            try:
                scene_frames_counter[key] += 1
            except KeyError:
                scene_frames_counter[key] = 1

        for key in list(scene_frames_counter.keys()):
            if scene_frames_counter[key] == self.dataset['images_per_root'] ** 2:
                self.completed_scenes.append(key)

        print('Define ' + str(len(self.completed_scenes)) + ' completed scenes\n')

    def set_completed_roots(self):
        print('Run set_completed_roots()')
        self.completed_roots = []

        root_frames_counter = {}

        for item in self.data:
            key = get_image_key_for_root(item)

            try:
                root_frames_counter[key] += 1
            except KeyError:
                root_frames_counter[key] = 1

        for key in list(root_frames_counter.keys()):
            if root_frames_counter[key] == self.dataset['images_per_root']:
                self.completed_roots.append(key)

        print('Define ' + str(len(self.completed_roots)) + ' completed roots\n')

    def set_completed_images(self):
        print('Run set_completed_images()')
        self.completed_images = []

        for image in self.data:
            self.completed_images.append(get_image_key_for_image(image))

        print('Define completed images count ' + str(len(self.completed_images)) + '\n')

    def is_image_calculated(self, image):
        # Check whole calculated scenes
        if get_image_key_for_scene(image) in self.completed_scenes:
            return True

        # Check whole calculated roots
        if get_image_key_for_root(image) in self.completed_roots:
            return True

        # Check image file personality
        if get_image_key_for_image(image) in self.completed_images:
            return True

        return False

    def save_data(self):
        f = open(self.file_path, 'w', encoding='utf-8')
        json.dump(self.data, f, ensure_ascii=False, indent=0, check_circular=False)
        f.close()


class DataCollector:
    def __init__(self, data_sources):
        for dataset in data_sources:
            DatasetCollector(dataset)


if __name__ == '__main__':
    DataCollector(DATA_SOURCES)
