import os
import re

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
        self.target_camera = target_camera
        self.frame_step = frame_step

        images_dir = project_name + '_images'
        self.save_directory_os = os.path.join(bpy.path.abspath('//'), '..', images_dir, 'cross', 'scene-'
                                              + str(self.target_camera))
        self.save_directory = '//..\\' + images_dir + '\\cross\\scene-' + str(self.target_camera) + '\\'

        exist_images = []

        files = os.listdir(self.save_directory_os)
        print('Walk in ' + self.save_directory_os)
        for file in files:
            if '.png' in file:
                path_to_image = os.path.join(self.save_directory_os, file)
                extract_data_from_path = re.findall(r'scene-(\d+).*root-(\d+).*frame-(\d+)', path_to_image)

                exist_images.append({
                    'scene': int(extract_data_from_path[0][0]),
                    'root': int(extract_data_from_path[0][1]),
                    'frame': int(extract_data_from_path[0][2]),
                })

        print('Start renders')
        for i in self.closed_range(root_frame_start, root_frame_end):
            for j in self.closed_range(frame_start, frame_end, self.frame_step):
                exist_image_filter = lambda v: v['root'] == i and v['frame'] == j
                exist_image = next(filter(exist_image_filter, exist_images), None)

                print('Camera: ' + str(self.target_camera) + '. Root ' + str(i) + '. Frame: ' + str(
                    j) + '. Image exist: ' + str(
                    exist_image))

                if not exist_image:
                    # reset frame position
                    self.setup_render_params(i, j)
                    bpy.context.scene.frame_set(j)

                    # render
                    bpy.ops.render.render(animation=False, write_still=True)

    def closed_range(self, start, stop, step=1):
        dir = 1 if (step > 0) else -1
        return range(start, stop + dir, step)

    def setup_render_params(self, root_frame, frame):
        bpy.context.scene.frame_set(root_frame)
        bpy.data.scenes['Scene'].frame_start = ((frame - 1) % self.frame_step) + 1

        box = bpy.data.objects['LightSceneBoxCamera' + str(self.target_camera)]
        cam = bpy.data.objects['CameraScene' + str(self.target_camera)]

        box.location = cam.location
        box.rotation_euler = cam.rotation_euler

        path_to_render_output = self.save_directory + 'root-' + str(root_frame) + '_frame-' + str(frame)
        bpy.data.scenes['Scene'].render.filepath = path_to_render_output
