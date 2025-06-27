import os
import shutil

def copy_png_files(source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.endswith('.png'):
            idx = int(filename.split('.')[0].split('_')[-1]) * 5
            source_file = os.path.join(source_dir, filename)
            destination_file = os.path.join(destination_dir, "maskraw_"+str(idx) + '.png')
            shutil.copy2(source_file, destination_file)
            print(f'Copied: {source_file} to {destination_file}')

with open('/YourPath/scannet_scene_val.txt', 'r') as file:
    data_list = [line.strip() for line in file.readlines()]
video_dir_scene_ids = data_list
base_dir = '/Yoursegtrackoutput/outputs'
video_dir_scene_ids = os.listdir(base_dir)
for scene in video_dir_scene_ids:
    source_directory = os.path.join(base_dir, scene, 'mask_data_merge')
    if not os.path.exists(source_directory):
        continue
    destination_directory = os.path.join('/YourScanNetPath/2D_masks', scene, 'semantic-sam')
    if not os.path.exists(destination_directory):
        copy_png_files(source_directory, destination_directory)
