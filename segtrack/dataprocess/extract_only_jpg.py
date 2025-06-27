import os
import shutil
opj = os.path.join
ol = os.listdir
path_list = ['/YourPath/ScanNet']
save_path = '/YourPath/ScanNet/color_images_cluster'
os.makedirs(save_path,exist_ok=True)

frame_step = 5  
for path in path_list:
    for scene_dir in ol(path):
        if os.path.exists(opj(save_path, scene_dir)):
            continue
        
        os.makedirs(opj(save_path, scene_dir), exist_ok=True)
        for img in ol(opj(path, scene_dir, 'color')):
            file_prefix = int(img.split('.')[0])
            if file_prefix % frame_step != 0:
                continue
            shutil.copyfile(opj(path, scene_dir, 'color', img), opj(save_path, scene_dir, str(int(file_prefix / frame_step))+'.jpg'))
