# [CVPR 2025] SAM2Object: Consolidating View Consistency via SAM2 for Zero-Shot 3D Instance Segmentation

 [Jihuai Zhao](https://jihuaizhaohd.github.io/), [Junbao Zhuo<sup>*</sup>](https://scholar.google.com/citations?user=iBt9uHUAAAAJ) [Jiansheng Chen](https://scholar.google.com/citations?user=A1gA9XIAAAAJ), [Huimin Ma<sup>*</sup>](https://scholar.google.com.hk/citations?user=32hwVLEAAAAJ&hl)

University of Science and Technology Beijing &nbsp; &nbsp;


<!-- **CVPR 2025** -->

[Project Page](https://jihuaizhaohd.github.io/SAM2Object/) | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_SAM2Object_Consolidating_View_Consistency_via_SAM2_for_Zero-Shot_3D_Instance_CVPR_2025_paper.html)

## Introduction

In the field of zero-shot 3D instance segmentation, existing 2D-to-3D lifting methods typically obtain 2D segmentation across multiple RGB frames using vision foundation models, which are then projected and merged into 3D space. However, since the inference of vision foundation models on a single frame is not integrated with adjacent frames, the masks of the same object may vary across different frames, leading to a lack of view consistency in the 2D segmentation. Furthermore, current lifting methods average the 2D segmentation from multiple views during the projection into 3D space, causing low-quality masks and high-quality masks to share the same weight. These factors can lead to fragmented 3D segmentation. 

<img src="assets\overview_.png" style="zoom: 33%;" />

We present SAM2Object, a novel zero-shot 3D instance segmentation method that effectively utilizes the Segment Anything Model 2 to segment and track objects, consolidating view consistency across frames. Our approach combines these consistent 2D masks with 3D geometric priors, improving the robustness of 3D segmentation. Additionally, we introduce mask consolidation module to filter out low-quality masks across frames, which enables more precise 3D-to-2D matching. Comprehensive evaluations on ScanNetV2, ScanNet++ and ScanNet200 demonstrate the robustness and effectiveness of SAM2Object, showcasing its ability to outperform previous methods.

## Usage

### Installation

Prepare environment:

```bash
conda create -n sam2object python=3.8
conda activate sam2object
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install open3d natsort matplotlib tqdm opencv-python scipy plyfile
# Install SAM2
cd segtrack
pip install -e .
```

Download the pretrained SAM 2 checkpoints:
```bash
cd segtrack/checkpoints
bash download_ckpts.sh
```




### Data Preparation

#### ScanNet
Download [ScanNetV2 / ScanNet200](https://github.com/ScanNet/ScanNet) and organize the dataset as follows:
 
```
data
 ├── ScanNet
 │   ├── posed_images
 │   |   ├── scene0000_00
 │   |   │   ├──intrinsic_color.txt   
 │   |   │   ├──intrinsic_depth.txt   
 │   |   │   ├──0000.jpg     //rgb image
 │   |   │   ├──0000.png     //depth image
 │   |   │   ├──0000.txt     //extrinsic
 │   |   │   └── ...
 │   |   └── ...
 │   ├── scans
 │   |   ├── scene0000_00
 │   |   └── ...
 │   ├── color_images_cluster
 │   |   ├── scene0000_00
 │   |   └── ...
 │   ├── Tasks
 │   |   ├── Benchmark
 │   |   │   ├──scannetv2_val.txt  
 │   |   │   ├──scannetv2_train.txt  
 │   |   │   └── ...
```

Generate the `color_images_cluster` , which contains the processed RGB image sequences derived from 20% of the ScanNet.
```bash
cd segtrack
python dataprocess/extract_only_jpg.py
```
Generate the `posed_images`
```bash
cd segtrack
python dataprocess/get_posed_images.py
```

### Get class-agnostic masks
1. **Obtain 2D SAM2 results**
   
   run:
   ```bash
   cd segtrack
   python seg_tracking.py
   ```
   The results will be stored at `segtrack/outputs`. `result*` folders: Store the visualizations of the masks overlaid on the images. `mask*` folders: Store the raw mask files. The suffixes on these folders denote the tracking direction: `No suffix`: Results from forward tracking. `_rev` suffix: Results from backward tracking. `_merge` suffix: Merged results from bidirectional tracking.

2. **Obtain superpoints**
   For ScanNet, superpoints are already provided in `scans/<scene_id>/<scene_id>_vh_clean_2.0.010000.segs.json`

   To generate superpoint on mesh of other dataset, following [SAI3D](https://github.com/yd-yin/SAI3D/), we also use the mesh segmentator provided by ScanNet directly. Please check [here](https://github.com/ScanNet/ScanNet/tree/master/Segmentator) to see the usage.

3. **Final 3D instance segmentation**
   Get final 3D instance segmentation by using the following command:
   ```bash
   cd graphclustering
   bash scripts/seg_scannet.sh
   ```


### Citation
If you find our work helpful for your research, please consider citing our paper.
```BibTex
@inproceedings{zhao2025sam2object,
  title={SAM2Object: Consolidating View Consistency via SAM2 for Zero-Shot 3D Instance Segmentation},
  author={Zhao, Jihuai and Zhuo, Junbao and Chen, Jiansheng and Ma, Huimin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={19325--19334},
  year={2025}
}
```