import bpy


class AnimLiningUp:
    def __init__(self, frames_count=800, scene=1):
        self.frames_count = frames_count
        self.scene_num = scene

        self.cam = bpy.data.objects['CameraScene' + str(scene)]
        self.box = bpy.data.objects['LightSceneBoxCamera' + str(scene)]

        self.set_frames_limit()
        self.set_light_box_keyframes()

    def set_frames_limit(self):
        line_frames_count = self.frames_count ** 2
        bpy.data.scenes['Scene'].frame_end = line_frames_count

    def set_light_box_keyframes(self):
        root_frames = self.frames_count

        for i in range(root_frames):
            root_frame = i + 1
            print('=== Work with root ' + str(root_frame) + ' === \n')

            bpy.context.scene.frame_set(root_frame)

            cam_loc = self.cam.location
            cam_rot = self.cam.rotation_euler

            print('Camera location: ' + str(cam_loc))
            print('Camera rotation: ' + str(cam_rot))

            self.box.location = cam_loc
            self.box.rotation_euler = cam_rot

            box_anim_frame_target = i * self.frames_count + 1
            box_anim_frame_before_next = box_anim_frame_target + self.frames_count - 1

            print('Apply box animation for frame: ' + str(box_anim_frame_target))
            print('And before next frame: ' + str(box_anim_frame_before_next))

            # Target frame
            self.box.keyframe_insert(data_path='location', frame=box_anim_frame_target)
            self.box.keyframe_insert(data_path='rotation_euler', frame=box_anim_frame_target)

            # Before next frame
            self.box.keyframe_insert(data_path='location', frame=box_anim_frame_before_next)
            self.box.keyframe_insert(data_path='rotation_euler', frame=box_anim_frame_before_next)
