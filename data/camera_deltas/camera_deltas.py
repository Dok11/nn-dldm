import bpy
import json
import math


######################################################################
#
# Script for Blender 2.81
#
# For exporting camera positions from blend files
# in absolute unit values for scene.
# Angle save in radians, distance in meter, FOV in degrees.
#
# Code to run it in Blender:
# import bpy
# import os
#
# dir = os.path.dirname(bpy.data.filepath)
# filename = os.path.join(dir, "..", "camera_deltas.py")
# exec(compile(open(filename).read(), filename, 'exec'))
#
######################################################################

CAMERA_COUNT = 10
SCENE = bpy.data.scenes['Scene']

FRAME_START = SCENE.frame_start
FRAME_END = SCENE.frame_end

RAD_TO_DEGREE = 57.2958  # 180/pi


def get_camera_rot_q(camera):
    q = camera.rotation_euler.to_quaternion()

    return {
        'w': q.w,
        'x': q.x,
        'y': q.y,
        'z': q.z,
    }


def get_camera_rot_e(camera):
    q = camera.rotation_euler

    return {
        'x': q.x,
        'y': q.y,
        'z': q.z,
    }


def get_camera_pos():
    camera_data = []

    # reset frame position
    bpy.context.scene.frame_set(FRAME_START)

    # save current camera
    first_camera = bpy.data.objects['CameraScene1']

    # save cam positions for every frame
    for frame_num in range(FRAME_START, FRAME_END + 1):
        bpy.context.scene.frame_set(frame_num)

        camera_rot_q = get_camera_rot_q(first_camera)
        camera_rot_e = get_camera_rot_e(first_camera)

        camera_data.append({
            'frame': bpy.context.scene.frame_current,
            'loc': {
                'x': first_camera.location.x,
                'y': first_camera.location.y,
                'z': first_camera.location.z,
            },
            'rot_q': {
                'w': camera_rot_q['w'],
                'x': camera_rot_q['x'],
                'y': camera_rot_q['y'],
                'z': camera_rot_q['z'],
            },
            'rot_e': {
                'x': camera_rot_e['x'],
                'y': camera_rot_e['y'],
                'z': camera_rot_e['z'],
            },
        })

    return camera_data


def get_data():
    camera_positions = get_camera_pos()

    # Work with every scene/camera (every scene have own camera)
    data = []
    for i in range(CAMERA_COUNT):
        scene_number = i + 1
        print('scene number', scene_number)
        camera = bpy.data.objects['CameraScene' + str(scene_number)].data
        angle = camera.angle * RAD_TO_DEGREE

        # Work with camera positions (every camera have some movements)
        for j in range(len(camera_positions)):
            data_item = camera_positions[j].copy()
            data_item['fov'] = round(angle, 2)
            data_item['scene'] = scene_number
            data_item['image_name'] = 'scene' + str(scene_number) + '_' + str(data_item['frame']) + '.jpg'

            data.append(data_item)

    return data


# Save data
filepath = bpy.path.abspath('//data.json')
f = open(filepath, 'w', encoding='utf-8')
json.dump(get_data(), f, ensure_ascii=False, indent=2)
f.close()
