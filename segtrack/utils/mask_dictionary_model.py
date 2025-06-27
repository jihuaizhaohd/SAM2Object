import numpy as np
import json
import torch
import copy
import os
import cv2
from dataclasses import dataclass, field

@dataclass
class MaskDictionaryModel:
    mask_name:str = ""
    mask_height: int = 1080 # 968
    mask_width:int = 1920 # 1296
    promote_type:str = "mask"
    labels:dict = field(default_factory=dict)


    def add_new_frame_annotation(self, mask_list, background_value = 0):
        # import pdb; pdb.set_trace()
        mask_img = torch.zeros(mask_list.shape[-2:])
        anno_2d = {}
        # import pdb; pdb.set_trace()
        for idx, (mask) in enumerate(mask_list):
            final_index = background_value + idx + 1

            if mask.shape[0] != mask_img.shape[0] or mask.shape[1] != mask_img.shape[1]:
                raise ValueError("The mask shape should be the same as the mask_img shape.")
            # mask = mask
            mask_img[mask == True] = final_index
            # print("label", label)
            new_annotation = ObjectInfo(instance_id = final_index, mask = mask)
            anno_2d[final_index] = new_annotation

        # np.save(os.path.join(output_dir, output_file_name), mask_img.numpy().astype(np.uint16))
        self.mask_height = mask_img.shape[0]
        self.mask_width = mask_img.shape[1]
        self.labels = anno_2d

    # def sorted_label(self):
        # self.labels = {k: v for k, v in sorted(self.labels.items(), key=lambda x: x[1]['predicted_iou'])}

    def add_new_frame_annotation_sort(self, image, mask_list, background_value = 0):
        # import pdb; pdb.set_trace()
        # sorted_masks = sorted(mask_list, key=lambda x: x['area'],reverse=True)
        sorted_masks = sorted(mask_list, key=(lambda x: x["predicted_iou"]))
        mask_img = torch.zeros(image.shape[:2])
        anno_2d = {}
        # import pdb; pdb.set_trace()
        for idx, (maskitem) in enumerate(sorted_masks):
            final_index = background_value + idx + 1
            mask = torch.from_numpy(maskitem['segmentation'])
            if mask.shape[0] != mask_img.shape[0] or mask.shape[1] != mask_img.shape[1]:
                raise ValueError("The mask shape should be the same as the mask_img shape.")
            # mask = mask
            mask_img[mask == True] = final_index
            # print("label", label)
            new_annotation = ObjectInfo(instance_id = final_index, mask = mask, predicted_iou=maskitem['area'])
            anno_2d[final_index] = new_annotation

        # np.save(os.path.join(output_dir, output_file_name), mask_img.numpy().astype(np.uint16))
        self.mask_height = mask_img.shape[0]
        self.mask_width = mask_img.shape[1]
        self.labels = anno_2d


    def add_new_frame_annotation_rev(self, sorted_mask, mask_list, background_value = 0):
        anno_2d = {}
        max_idx = int(mask_list.max())
        # import pdb; pdb.set_trace()
        v,counts = torch.unique(mask_list, return_counts=True)
        for i in v:
            if i==0:
                continue
            # i = int(i)
            mask = (mask_list == i)
            result_array = torch.zeros_like(mask_list)
            result_array[mask] = i
            new_annotation = ObjectInfo(instance_id = i, mask = result_array, predicted_iou=sorted_mask[int(i)]['predicted_iou'])
            anno_2d[int(i)] = new_annotation

        # np.save(os.path.join(output_dir, output_file_name), mask_list.numpy().astype(np.uint16))
        self.mask_height = mask_list.shape[0]
        self.mask_width = mask_list.shape[1]
        self.labels = anno_2d

    # ori
    # def add_new_frame_annotation(self, mask_list, box_list, label_list, background_value = 0):
    #     mask_img = torch.zeros(mask_list.shape[-2:])
    #     anno_2d = {}
    #     for idx, (mask, box, label) in enumerate(zip(mask_list, box_list, label_list)):
    #         final_index = background_value + idx + 1

    #         if mask.shape[0] != mask_img.shape[0] or mask.shape[1] != mask_img.shape[1]:
    #             raise ValueError("The mask shape should be the same as the mask_img shape.")
    #         # mask = mask
    #         mask_img[mask == True] = final_index
    #         # print("label", label)
    #         name = label
    #         box = box # .numpy().tolist()
    #         new_annotation = ObjectInfo(instance_id = final_index, mask = mask, class_name = name, x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3])
    #         anno_2d[final_index] = new_annotation

    #     # np.save(os.path.join(output_dir, output_file_name), mask_img.numpy().astype(np.uint16))
    #     self.mask_height = mask_img.shape[0]
    #     self.mask_width = mask_img.shape[1]
    #     self.labels = anno_2d

    # def update_masks(self, tracking_annotation_dict, iou_threshold=0.8, objects_count=0):
    #     updated_masks = {}

    #     for seg_obj_id, seg_mask in self.labels.items():  # tracking_masks
    #         flag = 0 
    #         new_mask_copy = ObjectInfo()
    #         if seg_mask.mask.sum() == 0:
    #             continue
            
    #         for object_id, object_info in tracking_annotation_dict.labels.items(): 
    #             iou = self.calculate_iou(seg_mask.mask, object_info.mask)  # tensor, numpy
    #             iou_o = self.calculate_iou_object(seg_mask.mask, object_info.mask)  # tensor, numpy
    #             iou_t = self.calculate_iou_track(seg_mask.mask, object_info.mask)  # tensor, numpy
    #             # 高 iou_o，低 iou_t, 新分割被包含在track里，是trak分割的一小部分, 用track替代 Partial-object Segmentation
    #             # 低 iou_o，高 iou_t, track被包含在新分割里，是新分割的一小部分, 用新分割替代 Oversized Segmentation 
    #             # 低 iou_o，低 iou_t, 两者属于交叉 {又分为和谁交叉的多， iou_o > iou_t, 与iou_o交叉的多； iou_o < iou_t, 与iou_t交叉的多} 视为 新物体

    #             # print("iou", iou)
    #             if iou > iou_threshold: # Complete-object Segmentation
    #                 # old
    #                 # flag = object_info.instance_id
    #                 # new_mask_copy.mask = seg_mask.mask
    #                 # new_mask_copy.instance_id = object_info.instance_id

    #                 # new
    #                 flag = object_info.instance_id
    #                 new_mask_copy.mask = object_info.mask
    #                 new_mask_copy.instance_id = object_info.instance_id
    #                 new_mask_copy.predicted_iou = object_info.predicted_iou
    #                 # new_mask_copy.class_name = seg_mask.class_name
    #                 break
    #             else:
    #             # 高 iou_o，低 iou_t, 新分割被包含在track里，是trak分割的一小部分, 用track替代 Partial-object Segmentation
    #                 if iou_o > 0.8 and iou_t < 0.4:
    #                     flag = object_info.instance_id
    #                     new_mask_copy.mask = object_info.mask
    #                     new_mask_copy.instance_id = object_info.instance_id
    #                     new_mask_copy.predicted_iou = object_info.predicted_iou
    #                     break
    #                 elif iou_t > 0.8 and iou_o < 0.4:
    #                     flag = object_info.instance_id
    #                     new_mask_copy.mask = seg_mask.mask
    #                     new_mask_copy.instance_id = object_info.instance_id
    #                     new_mask_copy.predicted_iou = seg_mask.predicted_iou



    #         if not flag:
    #             objects_count += 1
    #             flag = objects_count
    #             new_mask_copy.instance_id = objects_count
    #             new_mask_copy.mask = seg_mask.mask
    #             new_mask_copy.predicted_iou = seg_mask.predicted_iou

    #             # new_mask_copy.class_name = seg_mask.class_name
    #         updated_masks[flag] = new_mask_copy
    #     self.labels = updated_masks
    #     return objects_count



    def update_masks(self, tracking_annotation_dict, iou_threshold=0.8, objects_count=0):
        updated_masks = {}

        for seg_obj_id, seg_mask in self.labels.items():  # tracking_masks
            flag = 0 
            new_mask_copy = ObjectInfo()
            if seg_mask.mask.sum() == 0:
                continue

            objects_count += 1
            flag = objects_count
            new_mask_copy.instance_id = objects_count
            new_mask_copy.mask = seg_mask.mask
            new_mask_copy.predicted_iou = seg_mask.predicted_iou

                # new_mask_copy.class_name = seg_mask.class_name
            updated_masks[flag] = new_mask_copy
        self.labels = updated_masks
        return objects_count



    # def update_masks(self, tracking_annotation_dict, iou_threshold=0.8, objects_count=0):
    #     updated_masks = {}

    #     for seg_obj_id, seg_mask in self.labels.items():  # tracking_masks
    #         flag = 0 
    #         new_mask_copy = ObjectInfo()
    #         if seg_mask.mask.sum() == 0:
    #             continue
            
    #         for object_id, object_info in tracking_annotation_dict.labels.items(): 
    #             iou = self.calculate_iou(seg_mask.mask, object_info.mask)  # tensor, numpy
    #             iou_o = self.calculate_iou_object(seg_mask.mask, object_info.mask)  # tensor, numpy
    #             iou_t = self.calculate_iou_track(seg_mask.mask, object_info.mask)  # tensor, numpy
    #             # 高 iou_o，低 iou_t, 新分割被包含在track里，是trak分割的一小部分, 用track替代 Partial-object Segmentation
    #             # 低 iou_o，高 iou_t, track被包含在新分割里，是新分割的一小部分, 用新分割替代 Oversized Segmentation 
    #             # 低 iou_o，低 iou_t, 两者属于交叉 {又分为和谁交叉的多， iou_o > iou_t, 与iou_o交叉的多； iou_o < iou_t, 与iou_t交叉的多} 视为 新物体


                
    #             # print("iou", iou)
    #             if iou > iou_threshold: # Complete-object Segmentation
    #                 # old
    #                 # flag = object_info.instance_id
    #                 # new_mask_copy.mask = seg_mask.mask
    #                 # new_mask_copy.instance_id = object_info.instance_id

    #                 # new
    #                 flag = object_info.instance_id
    #                 new_mask_copy.mask = object_info.mask
    #                 new_mask_copy.instance_id = object_info.instance_id
    #                 # new_mask_copy.class_name = seg_mask.class_name
    #                 break

                
    #         if not flag:
    #             objects_count += 1
    #             flag = objects_count
    #             new_mask_copy.instance_id = objects_count
    #             new_mask_copy.mask = seg_mask.mask
    #             # new_mask_copy.class_name = seg_mask.class_name
    #         updated_masks[flag] = new_mask_copy
    #     self.labels = updated_masks
    #     return objects_count

    def get_target_class_name(self, instance_id):
        return self.labels[instance_id].class_name

    def get_target_logit(self, instance_id):
        return self.labels[instance_id].logit
    
    @staticmethod
    def calculate_iou(mask1, mask2):
        # Convert masks to float tensors for calculations
        mask1 = mask1.to(torch.float32)
        mask2 = mask2.to(torch.float32)
        
        # Calculate intersection and union
        intersection = (mask1 * mask2).sum()
        union = mask1.sum() + mask2.sum() - intersection
        
        # Calculate IoU
        iou = intersection / union
        return iou
    
    @staticmethod
    def calculate_iou_object(mask1, mask2):
        # Convert masks to float tensors for calculations
        # mask1 新分割， mask2 跟踪
        mask1 = mask1.to(torch.float32)
        mask2 = mask2.to(torch.float32)


        # Calculate intersection and union
        intersection = (mask1 * mask2).sum()
        union = mask1.sum()
        
        # Calculate IoU
        iou = intersection / union
        return iou

    @staticmethod
    def calculate_iou_track(mask1, mask2):
        # Convert masks to float tensors for calculations
        # mask1 新分割， mask2 跟踪
        mask1 = mask1.to(torch.float32)
        mask2 = mask2.to(torch.float32)
        


        # Calculate intersection and union
        intersection = (mask1 * mask2).sum()
        union = mask2.sum()
        
        # Calculate IoU
        iou = intersection / union
        return iou

    def to_dict(self):
        return {
            "mask_name": self.mask_name,
            "mask_height": self.mask_height,
            "mask_width": self.mask_width,
            "promote_type": self.promote_type,
            "labels": {k: v.to_dict() for k, v in self.labels.items()}
        }
    
    def to_json(self, json_file):
        with open(json_file, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
            
    def from_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
            self.mask_name = data["mask_name"]
            self.mask_height = data["mask_height"]
            self.mask_width = data["mask_width"]
            self.promote_type = data["promote_type"]
            self.labels = {int(k): ObjectInfo(**v) for k, v in data["labels"].items()}
        return self


@dataclass
class ObjectInfo:
    instance_id:int = 0
    mask: any = None
    class_name:str = ""
    x1:int = 0
    y1:int = 0
    x2:int = 0
    y2:int = 0
    logit:float = 0.0
    predicted_iou:float = 0.0

    def get_mask(self):
        return self.mask
    
    def get_id(self):
        return self.instance_id

    def get_predicted_iou(self):
        return self.predicted_iou

    def update_box(self):
        # 找到所有非零值的索引
        nonzero_indices = torch.nonzero(self.mask)
        
        # 如果没有非零值，返回一个空的边界框
        if nonzero_indices.size(0) == 0:
            # print("nonzero_indices", nonzero_indices)
            return []
        
        # 计算最小和最大索引
        y_min, x_min = torch.min(nonzero_indices, dim=0)[0]
        y_max, x_max = torch.max(nonzero_indices, dim=0)[0]
        
        # 创建边界框 [x_min, y_min, x_max, y_max]
        bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]        
        self.x1 = bbox[0]
        self.y1 = bbox[1]
        self.x2 = bbox[2]
        self.y2 = bbox[3]
    
    def to_dict(self):
        return {
            "instance_id": self.instance_id,
            "class_name": self.class_name,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "logit": self.logit
            # "mask": self.mask.cpu().numpy().tolist()
        }