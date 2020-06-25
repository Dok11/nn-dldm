import os
import re
import hashlib

import bpy

######################################################################
#
# Script for Blender 2.81
#
# Script runs rendering images with combinations all possible camera
# position to each other. For example in scene with 800 frames we
# have 640 000 (800*800) combinations.
#
# Code to run it in Blender:
# import bpy
# import os
#
# dir = os.path.dirname(bpy.data.filepath)
# filename = os.path.join(dir, "..", "external_render.py")
# exec(compile(open(filename).read(), filename, 'exec'))
#
# ExternalRender(
#     project_name='archviz',
#     target_camera=1,
#     root_frame_start=211,
#     root_frame_end=220,
#     frame_start=1,
#     frame_end=800,
#     frame_step=1,
# )
#
######################################################################


def get_hash_image(image):
    string = 's' + str(image['scene']) + 'r' + str(image['root']) + 'f' + str(image['frame'])
    return hashlib.md5(string.encode('utf-8')).hexdigest()


class ExternalRender:
    def __init__(self,
                 project_name='archviz',
                 target_camera=1,
                 root_frame_start=1,
                 root_frame_end=1,
                 frame_start=1,
                 frame_end=1,
                 frame_step=1,
                 ):
        self.project_name = project_name
        self.target_camera = target_camera
        self.frame_step = frame_step
        self.root_frame_start = root_frame_start
        self.root_frame_end = root_frame_end
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.exist_images = []
        self.save_directory_os = ''
        self.hash_list = []

        self.set_save_directory()
        self.set_exist_images()
        self.set_hash_list()
        self.render()

    def closed_range(self, start, stop, step=1):
        direction = 1 if (step > 0) else -1
        return range(start, stop + direction, step)

    def setup_render_params(self, root_frame, frame):
        bpy.context.scene.frame_set(root_frame)
        bpy.data.scenes['Scene'].frame_start = ((frame - 1) % self.frame_step) + 1

        box = bpy.data.objects['LightSceneBoxCamera' + str(self.target_camera)]
        cam = bpy.data.objects['CameraScene' + str(self.target_camera)]

        box.location = cam.location
        box.rotation_euler = cam.rotation_euler

        images_dir = self.project_name + '_images'
        save_directory = '//..\\' + images_dir + '\\cross\\scene-' + str(self.target_camera) + '\\'
        path_to_render_output = save_directory + 'root-' + str(root_frame) + '_frame-' + str(frame)
        bpy.data.scenes['Scene'].render.filepath = path_to_render_output

    def set_save_directory(self):
        images_dir = self.project_name + '_images'
        scene = 'scene-' + str(self.target_camera)

        self.save_directory_os = os.path.join(bpy.path.abspath('//'), '..', images_dir, 'cross', scene)

    def set_exist_images(self):
        files = os.listdir(self.save_directory_os)

        print('Walk in ' + self.save_directory_os)
        for file in files:
            if '.png' in file:
                path_to_image = os.path.join(self.save_directory_os, file)
                extract_data_from_path = re.findall(r'scene-(\d+).*root-(\d+).*frame-(\d+)', path_to_image)

                self.exist_images.append({
                    'scene': int(extract_data_from_path[0][0]),
                    'root': int(extract_data_from_path[0][1]),
                    'frame': int(extract_data_from_path[0][2]),
                })

    def set_hash_list(self):
        self.hash_list = []

        for image in self.exist_images:
            self.hash_list.append(get_hash_image(image))

        self.hash_list = set(self.hash_list)

    def find_image_in_hash_list(self, image):
        if get_hash_image(image) in self.hash_list:
            return True

        return False

    def render(self):
        print('Start renders')
        for i in self.closed_range(self.root_frame_start, self.root_frame_end):
            for j in self.closed_range(self.frame_start, self.frame_end, self.frame_step):
                image_data = {
                    'scene': self.target_camera,
                    'root': i,
                    'frame': j,
                }
                image_desc = 'Scene: ' + str(image_data['scene']) + '. Root: ' + str(i) + '. Frame: ' + str(j)

                if self.find_image_in_hash_list(image_data):
                    print('SKIP CALC FOR: ' + image_desc)
                    continue

                # reset frame position
                self.setup_render_params(i, j)
                bpy.context.scene.frame_set(j)

                # render
                bpy.ops.render.render(animation=False, write_still=True)
