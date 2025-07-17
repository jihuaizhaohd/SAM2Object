import os
import numpy as np
import json
import cv2
opj = os.path.join
class IDMapper:
    def __init__(self, initial_id=400):
        self.mapping = {}
        self.current_id = initial_id

    def map_id(self, original_id):
        if original_id in self.mapping:
            return self.mapping[original_id]
        
        mapped_id = self.current_id + 1
        self.mapping[original_id] = mapped_id
        self.current_id += 1

        return mapped_id

    def get_mapping(self):
        return self.mapping


def merge_two_video(mask_data_dir, mask_data_dir_rev, key_list):
    mergesave_mask_dir = os.path.join(os.path.dirname(mask_data_dir), "mask_data_merge")
    mergesave_json_dir = os.path.join(os.path.dirname(mask_data_dir), "json_data_merge")

    os.makedirs(mergesave_mask_dir, exist_ok=True)
    os.makedirs(mergesave_json_dir, exist_ok=True)

    mask_order_data_list = []
    # key_list = [0,35,91]


    mask_data_path_list = [
        p for p in os.listdir(mask_data_dir)
        if os.path.splitext(p)[-1] in [".npy"]
    ]
    mask_data_path_list.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))

    mask_data_path_list_rev = [
        p for p in os.listdir(mask_data_dir_rev)
        if os.path.splitext(p)[-1] in [".npy"]
    ]
    mask_data_path_list_rev.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))

    for file in mask_data_path_list:
        if file.endswith('npy'):
            mask_tmp = np.load(os.path.join(mask_data_dir, file))
            mask_order_data_list.append(mask_tmp)

    mask_reorder_data_list = []
    for file in mask_data_path_list_rev:
        if file.endswith('npy'):
            mask_tmp = np.load(os.path.join(mask_data_dir_rev, file))
            mask_reorder_data_list.append(mask_tmp)

    mask_order_data= np.stack(mask_order_data_list, axis=0)
    mask_reorder_data= np.stack(mask_reorder_data_list, axis=0)

    max_idx = int(mask_order_data.max())


    id_mapper = IDMapper(initial_id=max_idx)

    for idx in range(len(key_list)):
        if key_list[idx] == 0:
            continue
        index = key_list[idx]
        
        for in_idx in range(index, key_list[idx-1]-1, -1):

        
            order_frame = mask_order_data[in_idx]
            reorder_frame = mask_reorder_data[in_idx]

            zero_indices = np.where(order_frame == 0)
            reorder_frame_uniquevalue_with_order_frame = np.unique(reorder_frame[zero_indices])
            
            region_order_zero = (order_frame == 0)
            flag = 0
            for u_v in reorder_frame_uniquevalue_with_order_frame:
                if u_v == 0:
                    continue
                region_reorder_uvalue = (reorder_frame == u_v)
                uint8_array = region_reorder_uvalue.astype(np.uint8) * 255
                intersection = np.logical_and(region_order_zero, region_reorder_uvalue)
                intersection_area = np.sum(intersection)
                union_area = np.sum(region_reorder_uvalue)
                union_area_bk = np.sum(region_order_zero)
                iou = intersection_area / union_area if union_area > 0 else 0
                iou_bk = intersection_area / union_area_bk if union_area_bk > 0 else 0
                if iou > 0.7: 
                # if iou > 0.7 or iou_bk > 0.7:
                    # merge
                    flag = 1
                    reorder_indices = np.where(reorder_frame == u_v)
                    order_frame[reorder_indices] = id_mapper.map_id(u_v)

            
            
            np.save(os.path.join(mergesave_mask_dir, "mask_"+str(in_idx)+'.npy'), order_frame)
            cv2.imwrite(os.path.join(
                mergesave_mask_dir, f'maskraw_{in_idx}.png'), order_frame)
        
            u_values = np.unique(order_frame)
            tmp_dict = {}
            for u_value in u_values:
                if u_value == 0:
                    continue
                t_dict = {}
                t_dict["instance_id"] = int(u_value)
                t_dict["class_name"] = ""
                t_dict["x1"] = 0
                t_dict["y1"] = 0
                t_dict["x2"] = 0
                t_dict["y2"] = 0
                t_dict["logit"] = 0.0

                tmp_dict[str(u_value)] = t_dict

            dict_tmp = {
                "mask_name": "mask_"+str(in_idx)+'.npy',
                "mask_height": 968,
                "mask_width": 1296,
                "promote_type": "mask",
                "labels": tmp_dict
            }
            json_data_path = os.path.join(mergesave_json_dir, "mask_"+str(in_idx)+'.npy'.replace(".npy", ".json"))
            with open(json_data_path, "w") as f:
                json.dump(dict_tmp, f)

                