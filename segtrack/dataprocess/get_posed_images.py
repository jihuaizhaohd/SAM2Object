import os
import shutil
opj = os.path.join
ol = os.listdir
path_list = ['/YourPath/ScanNet']
save_path = '/YourPath/ScanNet/posed_images'

for path in path_list:
    for scene_dir in ol(path):
        os.makedirs(opj(save_path, scene_dir), exist_ok=True)
        shutil.copyfile(opj(path, scene_dir, 'intrinsics', 'intrinsic_color.txt'), opj(save_path, scene_dir, 'intrinsics_color.txt'))
        shutil.copyfile(opj(path, scene_dir, 'intrinsics', 'intrinsic_depth.txt'), opj(save_path, scene_dir, 'intrinsics_depth.txt'))
        for img in ol(opj(path, scene_dir, 'color')):
            shutil.copyfile(opj(path, scene_dir, 'color', img), opj(save_path, scene_dir, img))

        for img in ol(opj(path, scene_dir, 'depth')):
            shutil.copyfile(opj(path, scene_dir, 'depth', img), opj(save_path, scene_dir, img))

        for img in ol(opj(path, scene_dir, 'pose')):
            shutil.copyfile(opj(path, scene_dir, 'pose', img), opj(save_path, scene_dir, img))
