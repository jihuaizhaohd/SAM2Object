import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from keyframe_extract_func import extract_keyframes
import json
import copy
import shutil
opj = os.path.join
ol = os.listdir


"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def opencv_showmask(image, mask):

    _, binary_mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

    colored_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    alpha = 0.5  
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask.astype(np.uint8), alpha, 0)
    return overlay

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def filter_mindiffnum(sorted_list, thres=10):

    i = 0
    while i < len(sorted_list) - 1:
        if i == len(sorted_list) - 2:
            break
        difference = sorted_list[i + 1] - sorted_list[i]
        if difference < thres:
            del sorted_list[i + 1]
        else:
            i += 1

    return sorted_list

def insert_average(sorted_list, thres=150):
    result = []
    for i in range(len(sorted_list) - 1):
        result.append(sorted_list[i])
        if sorted_list[i + 1] - sorted_list[i] > thres:
            average = int((sorted_list[i] + sorted_list[i + 1]) / 2)
            result.append(average)
    result.append(sorted_list[-1])  
    return result

sam2_checkpoint = "/Your_Path/SAM2Object/segtrack/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
video_predictor_rev = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
image_predictor = SAM2ImagePredictor(sam2_image_model)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_image_model,
    points_per_side=64,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25.0,
    use_m2m=True,
    multimask_output=False,
)

with open('scannet_scene_val.txt', 'r') as file:
    data_list = [line.strip() for line in file.readlines()]
video_dir_scene_ids = data_list
for scene_id in video_dir_scene_ids:

    video_dir = opj("/Your_ScanNet_Path/color_images_cluster", scene_id)

    output_dir = opj("/Your_Path/SAM2Object/segtrack/outputs",scene_id)
    if os.path.exists(opj(output_dir, "result_merge")):
        continue
    CommonUtils.creat_dirs(output_dir)
    mask_data_dir = os.path.join(output_dir, "mask_data")
    json_data_dir = os.path.join(output_dir, "json_data")
    result_dir = os.path.join(output_dir, "result")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)

    mask_data_dir_rev = os.path.join(output_dir, "mask_data_rev")
    json_data_dir_rev = os.path.join(output_dir, "json_data_rev")
    result_dir_rev = os.path.join(output_dir, "result_rev")
    CommonUtils.creat_dirs(mask_data_dir_rev)
    CommonUtils.creat_dirs(json_data_dir_rev)
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # init video predictor state
    inference_state = video_predictor.init_state(video_path=video_dir, offload_video_to_cpu=True, async_loading_frames=True)
    inference_state_rev = video_predictor_rev.init_state(video_path=video_dir, reverse=True, offload_video_to_cpu=True, async_loading_frames=True)

    sam2_masks = MaskDictionaryModel()
    PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
    objects_count = 0

    """
    Step 2.0: Get key frames idx
    """ 
    keyframe_idx_list = extract_keyframes(video_dir, len_window=20)
    keyframe_idx_list.append(len(frame_names) - 1)
    keyframe_idx_list.insert(0, 0)
    keyframe_idx_list = filter_mindiffnum(keyframe_idx_list, thres=5)
    keyframe_idx_list = insert_average(keyframe_idx_list, thres=40)

    # list of differences
    differences = []
    for i in range(len(keyframe_idx_list) - 1):
        difference = keyframe_idx_list[i + 1] - keyframe_idx_list[i]
        differences.append(difference)

    os.makedirs(opj(output_dir, "key_list"),exist_ok=True)
    with open(opj(output_dir, "key_list", "key_list.txt"), 'w') as f:
        f.write(', '.join(map(str, keyframe_idx_list)))

    """
    Step 2.1: Prompt SAM image predictor to get the mask for all frames
    """
    video_segments_all = {}
    print("Total frames:", len(frame_names))
    for idx in range(len(keyframe_idx_list) - 1):
        print(keyframe_idx_list)
        start_frame_idx = keyframe_idx_list[idx]
        print("start_frame_idx", start_frame_idx)

        diff_step = differences[idx]

        img_path = os.path.join(video_dir, frame_names[start_frame_idx])
        image = Image.open(img_path)
        image_base_name = frame_names[start_frame_idx].split(".")[0]
        mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")
        imagergb = np.array(image.convert("RGB"))

        masks = mask_generator.generate(np.array(image.convert("RGB")))

        """
        Step 3: Register each object's positive points to video predictor
        """
        # If you are using point prompts, we uniformly sample positive points based on the mask
        if mask_dict.promote_type == "mask":
            mask_dict.add_new_frame_annotation_sort(imagergb,mask_list=masks)
        else:
            raise NotImplementedError("SAM 2 video predictor only support mask prompts")
        """
        Step 4: One Stage ： Propagate the video predictor to get the segmentation results for each frame
        """
        objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.7, objects_count=objects_count)
        print("objects_count", objects_count)
        video_predictor.reset_state(inference_state)
        if len(mask_dict.labels) == 0:
            print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
            continue
        video_predictor.reset_state(inference_state)

        for object_id, object_info in mask_dict.labels.items():
            frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                    inference_state,
                    start_frame_idx,
                    object_id,
                    object_info.mask,
                )

        # Propagate for a short interval every diff_step frames
        video_segments = {}  # output the following {step} frames tracking masks
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, 
                                                                                            max_frame_num_to_track=diff_step, 
                                                                                            start_frame_idx=start_frame_idx
                                                                                            ):
            frame_masks = MaskDictionaryModel()
            
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0).cpu() # .cpu().numpy()
                object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], predicted_iou=mask_dict.labels[out_obj_id].predicted_iou)
                object_info.update_box()
                frame_masks.labels[out_obj_id] = object_info
                image_base_name = frame_names[out_frame_idx].split(".")[0]
                frame_masks.mask_name = f"mask_{image_base_name}.npy"
                frame_masks.mask_height = out_mask.shape[-2]
                frame_masks.mask_width = out_mask.shape[-1]

            video_segments[out_frame_idx] = frame_masks
            if out_frame_idx in keyframe_idx_list:
                video_segments_all[out_frame_idx] = frame_masks
            sam2_masks = copy.deepcopy(frame_masks)

        print("video_segments:", len(video_segments))

        """
        Step 5: One stage save the tracking masks and json files
        """
        
        for frame_idx, frame_masks_info in video_segments.items():
            mask = frame_masks_info.labels
            mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
            #sorted iou
            sorted_iou_dict = {}
            for obj_id, obj_info in mask.items():
                sorted_iou_dict[obj_id] = {}
                sorted_iou_dict[obj_id]['predicted_iou'] = obj_info.predicted_iou
                sorted_iou_dict[obj_id]['mask'] = obj_info.mask

            sorted_mask = {k: v for k, v in sorted(sorted_iou_dict.items(), key=lambda x: x[1]['predicted_iou'])}
            for obj_id, obj_info in sorted_mask.items():
                mask_img[obj_info['mask'] == True] = obj_id
            mask_img = mask_img.numpy().astype(np.uint16)
            np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)
            cv2.imwrite(os.path.join(
                mask_data_dir, f'maskraw_{frame_idx}.png'), mask_img)
            
            json_data = frame_masks_info.to_dict()
            json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
            with open(json_data_path, "w") as f:
                json.dump(json_data, f)

    """
    Step : two stage 
    """
    del sam2_masks
    del mask_dict
    del video_segments


    frame_names_rev = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names_rev.sort(key=lambda p: int(os.path.splitext(p)[0]), reverse=True)

    differences_rev = copy.deepcopy(differences)
    differences_rev.reverse()

    sam2_masks_rev = MaskDictionaryModel()

    print("Total frames:", len(frame_names_rev))
    # In the reverse order, bypass the step-by-step propagation. 
    video_predictor_rev.reset_state(inference_state_rev)
    diff_idx = 0
    frame_idx_list = []
    frame_masks_info_list = []
    for frame_idx, frame_masks_info in video_segments_all.items():
        frame_idx_list.insert(0,frame_idx)
        frame_masks_info_list.insert(0,frame_masks_info)

    for frame_idx, frame_masks_info in zip(frame_idx_list[:-1], frame_masks_info_list[:-1]):
        # video_predictor_rev.reset_state(inference_state_rev)
        print("start_frame_idx", frame_idx)
        start_frame_idx = frame_idx
        if start_frame_idx in keyframe_idx_list:
            image_base_name = frame_names[start_frame_idx].split(".")[0]
            mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")
            if diff_idx != 0:
                mask = frame_masks_info.labels
                masks = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
                #sorted iou
                sorted_iou_dict = {}
                for obj_id, obj_info in mask.items():
                    sorted_iou_dict[obj_id] = {}
                    sorted_iou_dict[obj_id]['predicted_iou'] = obj_info.predicted_iou
                    sorted_iou_dict[obj_id]['mask'] = obj_info.mask

                sorted_mask = {k: v for k, v in sorted(sorted_iou_dict.items(), key=lambda x: x[1]['predicted_iou'])}
                # sorted_mask = {k: v for k, v in sorted(sorted_iou_dict.items(), key=lambda x: x[1]['area'],reverse=True)}
                # import pdb; pdb.set_trace()
                for obj_id, obj_info in sorted_mask.items():
                    masks[obj_info['mask'] == True] = obj_id

            else:
                img_path = os.path.join(video_dir, frame_names[frame_idx])
                image = Image.open(img_path)
                imagergb = np.array(image.convert("RGB"))
                masks = mask_generator.generate(imagergb)
                # segmentation_arrays = [item['segmentation'] for item in masks]
                # segmentation_3d_array = np.array(segmentation_arrays)
                # masks = segmentation_3d_array.astype(np.float32)

            # If you are using point prompts, we uniformly sample positive points based on the mask
            if mask_dict.promote_type == "mask":
                if diff_idx != 0:
                    mask_dict.add_new_frame_annotation_rev(sorted_mask,mask_list=torch.tensor(masks).to('cpu'))
                else:
                    mask_dict.add_new_frame_annotation_sort(imagergb,mask_list=masks)
                # mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
            else:
                raise NotImplementedError("SAM 2 video predictor only support mask prompts")
            objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks_rev, iou_threshold=0.7, objects_count=objects_count)
            video_predictor_rev.reset_state(inference_state_rev)
            
            print("objects_count", objects_count)
            
            video_predictor_rev.reset_state(inference_state_rev)

            for object_id, object_info in mask_dict.labels.items():
                frame_idx, out_obj_ids, out_mask_logits = video_predictor_rev.add_new_mask(
                        inference_state_rev,
                        len(frame_names_rev)-1-start_frame_idx, 
                        object_id,
                        object_info.mask,
                    )
            

            video_segments = {}  # output the following {step} frames tracking masks
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor_rev.propagate_in_video(inference_state_rev, 
                                                                                                    max_frame_num_to_track=differences_rev[diff_idx], 
                                                                                                    start_frame_idx=len(frame_names_rev)-1-start_frame_idx
                                                                                                ):
                frame_masks = MaskDictionaryModel()
                
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0).cpu() # .cpu().numpy()
                    object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], predicted_iou=mask_dict.labels[out_obj_id].predicted_iou)
                    object_info.update_box()
                    frame_masks.labels[out_obj_id] = object_info
                    image_base_name = frame_names_rev[out_frame_idx].split(".")[0]
                    frame_masks.mask_name = f"mask_{image_base_name}.npy"
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]

                video_segments[image_base_name] = frame_masks
                sam2_masks_rev = copy.deepcopy(frame_masks)
        
            diff_idx += 1
            for frame_idx, frame_masks_info in video_segments.items():
                mask = frame_masks_info.labels
                mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width) # 0 is background
                #sorted iou
                sorted_iou_dict = {}
                for obj_id, obj_info in mask.items():
                    sorted_iou_dict[obj_id] = {}
                    sorted_iou_dict[obj_id]['predicted_iou'] = obj_info.predicted_iou
                    sorted_iou_dict[obj_id]['mask'] = obj_info.mask

                sorted_mask = {k: v for k, v in sorted(sorted_iou_dict.items(), key=lambda x: x[1]['predicted_iou'])}
                for obj_id, obj_info in sorted_mask.items():
                    mask_img[obj_info['mask'] == True] = obj_id

                mask_img = mask_img.numpy().astype(np.uint16)
                np.save(os.path.join(mask_data_dir_rev, frame_masks_info.mask_name), mask_img)
                cv2.imwrite(os.path.join(
                    mask_data_dir_rev, f'maskraw_{frame_idx}.png'), mask_img)

                json_data = frame_masks_info.to_dict()
                json_data_path = os.path.join(json_data_dir_rev, frame_masks_info.mask_name.replace(".npy", ".json"))
                with open(json_data_path, "w") as f:
                    json.dump(json_data, f)

    CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)
    CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir_rev, json_data_dir_rev, result_dir_rev)


    '''
    Merge
    '''
    from merge_video import merge_two_video
    merge_two_video(mask_data_dir, mask_data_dir_rev, keyframe_idx_list)

    mergesave_mask_dir = os.path.join(os.path.dirname(mask_data_dir), "mask_data_merge")
    mergesave_json_dir = os.path.join(os.path.dirname(mask_data_dir), "json_data_merge")
    mergeresult_dir = os.path.join(os.path.dirname(mask_data_dir), "result_merge")
    os.makedirs(mergesave_mask_dir, exist_ok=True)
    os.makedirs(mergesave_json_dir, exist_ok=True)
    os.makedirs(mergeresult_dir, exist_ok=True)
    CommonUtils.draw_masks_and_box_with_supervision(video_dir, mergesave_mask_dir, mergesave_json_dir, mergeresult_dir)
