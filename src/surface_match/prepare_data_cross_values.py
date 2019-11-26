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


for dataset in DATA_SOURCES:
    filepath = os.path.join(CURRENT_DIR, '..', '..', 'data', 'surface_match', dataset['code'] + '.json')

    # TODO: Get exist file with calculated value from json file
    if os.path.exists(filepath):
        with open(filepath) as json_file:
            data = json.load(json_file)
    else:
        data = []

    # Collect folders with files
    scenes = []
    folders = os.listdir(dataset['images_dir'])
    for folder in folders:
        if os.path.isdir(os.path.join(dataset['images_dir'], folder)):
            scenes.append(folder)

    # Collect image files
    image_files = []
    for scene in scenes:
        match_scene = re.findall(r'-(\d+)', scene)

        images = os.listdir(os.path.join(dataset['images_dir'], scene))

        # TODO: remove [:5]
        for image in images[:5]:
            match = re.findall(r'root-(\d+)_frame-(\d+)', image)

            if match:
                image_files.append({
                    'scene': int(match_scene[0]),
                    'path': os.path.join(dataset['images_dir'], scene, image),
                    'root': int(match[0][0]),
                    'frame': int(match[0][1]),
                })

    # Calculate values for image
    for image_file in image_files:
        img = Image.open(image_file['path']).convert('L')
        arr = np.array(img)
        flatten_arr = [x for sublist in arr for x in sublist]

        pixels = len(flatten_arr)
        white_sum = sum(flatten_arr)
        value = white_sum / 255 / pixels

        data.append({
            'scene': 1,
            'root': image_file['root'],
            'frame': image_file['frame'],
            'value': value,
        })

    # Save data to json
    # TODO: change archviz to dataset name
    # f = open(filepath, 'w', encoding='utf-8')
    # json.dump(data, f, ensure_ascii=False, indent=2)
    # f.close()

    print(data)
