import os


class AnimLiningBack:
    def __init__(self, file, scene=0, frames_per_scene=800):
        self.file = file
        self.scene = scene
        self.frames_per_scene = frames_per_scene

        self.path = os.path.join(os.curdir, self.file + '_images', 'cross', 'scene-' + str(self.scene))
        self.file_list = os.listdir(self.path)

    def rename_images(self):
        source_image_num = 0

        for root in range(1, self.frames_per_scene + 1):
            for target in range(1, self.frames_per_scene + 1):
                source_image_num += 1
                source_file_name = str(source_image_num).zfill(4) + '.png'
                source_file_path = os.path.join(self.path, source_file_name)

                if source_file_name not in self.file_list:
                    print('File ' + source_file_name + ' not found')
                    continue

                new_file_name = 'root-' + str(root) + '_frame-' + str(target) + '.png'
                new_file_path = os.path.join(self.path, new_file_name)

                os.rename(source_file_path, new_file_path)


anim_lining_back = AnimLiningBack('barbershop', 1, 800)
anim_lining_back.rename_images()
