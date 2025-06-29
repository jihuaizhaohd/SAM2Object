from os.path import join, basename, dirname
import os
import glob
import json
from natsort import natsorted
import plyfile
import argparse
import cv2

from sam2object_base import *
from linetimer import CodeTimer


class ScanNet_SAM2OBJECT(SAM2OBJECTBase):
    def __init__(self, points, args):
        self.scannetpp = args.scannetpp
        super().__init__(points, args)

    def init_data(self, scene_id, base_dir, mask_name, need_semantic=False):
        """
        base_dir: root dir of dataset
        mask_name: which group of mask to use(fast-sam, sam-hq)
        need_semantic: whether to load semantic mask(gained from ovseg)
        """
        self.poses, self.color_intrinsics, self.depth_intrinsics, self.masks, self.depths, self.semantic_masks = \
            self.get_mask_data(base_dir, scene_id, mask_name, need_semantic, self.scannetpp)
        self.M = self.masks.shape[0]
        self.CH, self.CW = self.masks.shape[-2:]
        self.DH, self.DW = self.depths.shape[-2:]
        self.base_dir = base_dir
        self.scene_id = scene_id

    def get_mask_data(self, 
                      base_dir, 
                      scene_id, 
                      mask_name, 
                      need_semantic=False, 
                      scannetpp=False):
        # import pdb; pdb.set_trace()
        if scannetpp:
            rgb_dir = join(base_dir, 'posed_images', scene_id, 'rgb')
            color_list = natsorted(glob.glob(join(rgb_dir, '*.jpg')))
            color_shape = np.array(cv2.imread(color_list[0])).shape

            masks = []
            depths = []
            semantic_masks = []

            poses, intrinsics = utils.get_scannetpp_poses_and_intrinsics(
                base_dir, scene_id, self.view_freq)

            for color_path in tqdm(color_list, desc='Read 2D data'):
                color_name = basename(color_path)
                num = int(color_name[-9:-4])
                if num % self.view_freq != 0:
                    continue
                masks.append(utils.get_scannetpp_mask(
                    color_path, base_dir, scene_id, mask_name, color_shape=color_shape))
                depths.append(utils.get_scannetpp_depth(
                    color_path, color_shape=color_shape))

            poses = np.stack(poses, 0)  # (M, 4, 4)
            intrinsics = np.stack(intrinsics, 0)  # (M, 3, 3)
            masks = np.stack(masks, 0)  # (M, H, W)
            depths = np.stack(depths, 0)  # (M, H, W)

            return poses, intrinsics, intrinsics, masks, depths, None

        else:
            data_dir = join(base_dir, 'posed_images', scene_id)
            color_list = natsorted(glob.glob(join(data_dir, '*.jpg')))

            poses = []
            color_intrinsics = []
            depth_intrinsics = []
            masks = []
            depths = []
            semantic_masks = []

            for color_path in tqdm(color_list, desc='Read 2D data'):
                color_name = basename(color_path)
                num = int(color_name[-9:-4])
                if num % self.view_freq != 0:
                    continue
                poses.append(utils.get_scannet_pose(color_path))
                depths.append(utils.get_scannet_depth(color_path))
                color_intrinsic, depth_intrinsic = \
                    utils.get_scannet_color_and_depth_intrinsic(color_path)
                color_intrinsics.append(color_intrinsic)
                depth_intrinsics.append(depth_intrinsic)
                masks.append(utils.get_scannet_mask(color_path, base_dir, scene_id, mask_name))
                if need_semantic:
                    semantic_masks.append(utils.get_scannet_semantic_mask(
                        color_path, base_dir, scene_id, self.args.scannet200))

            poses = np.stack(poses, 0)  # (M, 4, 4)
            color_intrinsics = np.stack(color_intrinsics, 0)  # (M, 3, 3)
            depth_intrinsics = np.stack(depth_intrinsics, 0)  # (M, 3, 3)
            masks = np.stack(masks, 0)  # (M, H, W)
            depths = np.stack(depths, 0)  # (M, H, W)
            if need_semantic:
                semantic_masks = np.stack(semantic_masks, 0)  # (M, H, W)
            else:
                semantic_masks = None

            return poses, color_intrinsics, depth_intrinsics, masks, depths, semantic_masks

    def get_seg_data(self,
                     base_dir,
                     scene_id,
                     max_neighbor_distance,
                     seg_ids=None,
                     points_obj_labels_path=None,
                     k_graph=8,
                     point_level=False):
        """get information about primitives for region growing

        :param base_dir: root dir of dataset
        :param scene_id: id of scene
        :param max_neighbor_distance: max distance for searching seg neighbors. 
        :param seg_ids(N,): ids of superpoints which each point belongs to.
        :param points_obj_labels_path: path to the file that contains points and their superpoint ids/
        :param k_graph: parameter for kdtree search
        :param point_level: whether to use points as primitives
        :return: seg_ids(N,): ids of superpoints which each point belongs to.
                 seg_num: number of superpoints
                seg_members: dict, key is superpoint id, value is the ids of points that belong to this superpoint
                seg_neineighbors: (max_neighbor_distance, seg_num, seg_num), binary matrix, "True" indicating the logical distance between two superpoints is leq max_neighbor_distance    
        """
        # base_dir = '/datasets/ScanNet'
        # use points as primitives of region growing
        if point_level:
            seg_ids = np.arange(self.N, dtype=int)
            seg_num = self.N
            seg_members = seg_ids
            points_kdtree = scipy.spatial.KDTree(self.points)
            points_neighbors = points_kdtree.query(self.points, k_graph, workers=n_workers)[1]  # (n,k)
            self.seg_member_count = np.ones(self.N, dtype=int)

            return seg_ids, seg_num, seg_members, points_neighbors

        # use superpoints(oversegmentation) as primitives of region growing
        if seg_ids is None:
            if points_obj_labels_path is None:
                # load superpoint ids from json file
                if not self.scannetpp:
                    scene_seg_path = \
                        join('/YourScanNetPath', 'scans', scene_id, f'{scene_id}_vh_clean_2.0.010000.segs.json')
                    with open(scene_seg_path, 'r') as f:
                        seg_data = json.load(f)
                    seg_ids = np.array(seg_data['segIndices'])
                else:
                    ply_path = join('/YourScanNetPPPath/data', args.scene_id, 'scans', 'mesh_aligned_0.05.ply')
                    superpoint_path = \
                        join(base_dir, 'superpoints', scene_id, 'superpoint.pts') #segments.json
                    
                    seg_ids = utils.get_scannet_superpoints(ply_path,superpoint_path)
            else:
                seg_path = join(base_dir, 'scans', scene_id, 'results', points_obj_labels_path)
                seg_ids = np.loadtxt(seg_path)[:, 4].astype(int)

        # project ids of superpoints to consecutive natural numbers starting from 0
        seg_ids = utils.num_to_natural(seg_ids)


        unique_seg_ids, counts = np.unique(
            seg_ids, return_counts=True)  # from 0 to seg_num-1
        seg_num = unique_seg_ids.shape[0]

        # count member points of each superpoint
        seg_members = {}  # save as dict to lower memory cost
        for id in unique_seg_ids:
            seg_members[id] = np.where(seg_ids == id)[0]

        # collect spatial neighboring superpoints of each superpoint
        # 1. find neighboring points of each point
        points_kdtree = scipy.spatial.KDTree(self.points)
        points_neighbors = points_kdtree.query(
            self.points, k_graph, workers=n_workers)[1]  # (n,k)
        # 2. find directly neighboring superpoints of each superpoint with the help of point neighbors
        # binary matrix, "True" indicating the two superpoints are neighboring
        seg_direct_neighbors = np.zeros((seg_num, seg_num), dtype=bool)
        for id, members in seg_members.items():
            neighbors = points_neighbors[members]
            neighbor_seg_ids = seg_ids[neighbors]
            seg_direct_neighbors[id][neighbor_seg_ids] = 1
        seg_direct_neighbors[np.eye(seg_num, dtype=bool)] = 0  # exclude self
        # make neighboring matrix symmetric
        seg_direct_neighbors[seg_direct_neighbors.T] = 1

        # 3. find indirectly neighboring superpoints of each superpoint
        # zeroth dimension is "distance" of two superpoints
        seg_neineighbors = np.zeros(
            (max_neighbor_distance, seg_num, seg_num), dtype=bool)
        seg_neineighbors[0] = seg_direct_neighbors
        for i in range(1, max_neighbor_distance):  # to get neighbors with ditance leq i+1
            for seg_id in range(seg_num):
                last_layer_neighbors = seg_neineighbors[i - 1, seg_id]
                this_layer_neighbors = seg_neineighbors[i - 1, last_layer_neighbors].sum(0) > 0
                seg_neineighbors[i, seg_id] = this_layer_neighbors
            # exclude self
            seg_neineighbors[i, np.eye(seg_num, dtype=bool)] = 0
            # include closer neighbors
            seg_neineighbors[i, seg_neineighbors[i - 1]] = 1

        self.seg_member_count = counts
        return seg_ids, seg_num, seg_members, seg_neineighbors


def everything_seg(args):
    time_collection = {}
    with CodeTimer('Load points', dict_collect=time_collection):
        if args.scannetpp:
            ply_path = join('/YourPath/scannetpp/data/data', args.scene_id, 'scans', 'mesh_aligned_0.05.ply')
            os.makedirs(join(args.base_dir, 'scans', args.scene_id), exist_ok=True)
            points_path = join(dirname(join(args.base_dir, 'scans', args.scene_id, 'mesh_aligned_0.05.ply')), 'points.pts')
        else:
            ply_path = join('/YourScanNetPath/scans', args.scene_id, f'{args.scene_id}_vh_clean_2.ply')
            # ply_path = join(args.base_dir, 'scans', args.scene_id, f'{args.scene_id}_vh_clean_2.ply')
        # points_path = join(dirname(ply_path), 'points.pts')
            os.makedirs(join(args.base_dir, 'scans', args.scene_id), exist_ok=True)
            points_path = join(dirname(join(args.base_dir, 'scans', args.scene_id, f'{args.scene_id}_vh_clean_2.ply')), 'points.pts')

        if not os.path.exists(points_path):
            print('getting points from ply...')
            utils.get_points_from_ply(ply_path, points_path)


        points = np.loadtxt(points_path).astype(np.float32)
        print('points num:', points.shape[0])

    save_dir = join(args.base_dir, 'scans', args.scene_id, 'results')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    agent = ScanNet_SAM2OBJECT(points, args)

    with CodeTimer('Load images', dict_collect=time_collection):
        # 加载poses, intrinsics, masks, depths
        agent.init_data(args.scene_id, args.base_dir, args.mask_name)

    with CodeTimer('Assign instance labels', dict_collect=time_collection):
        labels_fine_global = agent.assign_label(points, 
                                                thres_connect=args.thres_connect, 
                                                vis_dis=args.thres_dis,
                                                max_neighbor_distance=args.max_neighbor_distance, 
                                                similar_metric=args.similar_metric)

    
def ransac_plane_seg(points, pred_ins, floor_id, scene_dist_thres):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    plane, inliers = point_cloud.segment_plane(distance_threshold=scene_dist_thres, ransac_n=3, num_iterations=1000)
    pred_ins[inliers] = floor_id

    return pred_ins

def merge_floor(pred_ins, floor_propose_ids, floor_id, scene_inter_thres):
    unique_pre_ins_ids = np.unique(pred_ins)
    for i in range(len(unique_pre_ins_ids)):
        if unique_pre_ins_ids[i] == -1:
            pre_instance_points_idx = np.where(pred_ins == unique_pre_ins_ids[i])[0]
            insection = np.isin(pre_instance_points_idx, floor_propose_ids) # the intersection between the floor and the predicted instance
            if sum(insection) > 0: 
                pred_ins[pre_instance_points_idx[insection]] = floor_id
            continue
        
        pre_instance_points_idx = np.where(pred_ins == unique_pre_ins_ids[i])[0]
        insection = sum(np.isin(pre_instance_points_idx, floor_propose_ids))  # the intersection between the floor and the predicted instance
     
        ratio = insection / len(pre_instance_points_idx)
        if ratio > scene_inter_thres:
            pred_ins[pre_instance_points_idx] = floor_id
            print(unique_pre_ins_ids[i])

    return pred_ins

def export_ids(filename, ids):
    if not os.path.exists(dirname(filename)):
        os.mkdir(dirname(filename))
    with open(filename, 'w') as f:
        for id in ids:
            f.write('%d\n' % id)


def export_merged_ids_for_eval(instance_ids, 
                               save_dir, 
                               args, 
                               res_name='None', 
                               label_ids_dir=None):
    """
    code credit: scannet
    Export 3d instance labels for scannet class agnostic instance evaluation
    For semantic instance evaluation if label_ids_dir is not None
    """
    os.makedirs(save_dir, exist_ok=True)

    confidences = np.ones_like(instance_ids)

    if label_ids_dir is None:
        label_ids = np.ones_like(instance_ids, dtype=int)
    else:
        label_ids = np.loadtxt(
            join(label_ids_dir, f'{args.scene_id}.txt')).astype(int)

    filename = join(save_dir, f'{args.scene_id}.txt')
    print(f'export {res_name} to {filename}')

    output_mask_path_relative = f'{args.scene_id}_pred_mask'
    name = os.path.splitext(os.path.basename(filename))[0]
    output_mask_path = os.path.join(
        os.path.dirname(filename), output_mask_path_relative)
    if not os.path.isdir(output_mask_path):
        os.mkdir(output_mask_path)
    insts = np.unique(instance_ids)
    zero_mask = np.zeros(shape=(instance_ids.shape[0]), dtype=np.int32)
    with open(filename, 'w') as f:
        for idx, inst_id in enumerate(insts):
            if inst_id == 0:  # 0 -> no instance for this vertex
                continue
            relative_output_mask_file = os.path.join(
                output_mask_path_relative, name + '_' + str(idx) + '.txt')
            output_mask_file = os.path.join(
                output_mask_path, name + '_' + str(idx) + '.txt')
            loc = np.where(instance_ids == inst_id)
            label_id = label_ids[loc[0][0]]
            confidence = confidences[loc[0][0]]
            f.write('%s %d %f\n' %
                    (relative_output_mask_file, label_id, confidence))
            # write mask
            mask = np.copy(zero_mask)
            mask[loc[0]] = 1
            export_ids(output_mask_file, mask)


def write_json(path, data):
    with open(path, "w") as f:
        f.write(json.dumps(data, indent=4))

def export_scannetpp_eval(inst_gt, 
                        inst_predsformat_out_dir,
                        args):
    os.makedirs(inst_predsformat_out_dir, exist_ok=True)

    scene_id = args.scene_id
    # create main txt file
    main_txt_file = join(inst_predsformat_out_dir, f'{args.scene_id}.txt')
    # get the unique and valid instance IDs in inst_gt 
    # (ignore invalid IDs)
    inst_ids = np.unique(inst_gt)
    inst_ids = inst_ids[inst_ids > 0]
    # main txt file lines
    main_txt_lines = []

    # create the dir for the instance masks
    inst_masks_dir = join(inst_predsformat_out_dir, 'predicted_masks')
    os.makedirs(inst_masks_dir, exist_ok=True)
    # inst_masks_dir.mkdir(parents=True, exist_ok=True)

    # for each instance
    for inst_ndx, inst_id in enumerate(tqdm(sorted(inst_ids))):
    # get the mask for the instance
        inst_mask = inst_gt == inst_id
        # get the semantic label for the instance
        inst_sem_label = sem_gt[inst_mask][0]
        # add a line to the main file with relative path
        # predicted_masks <semantic label> <confidence=1>
        mask_path_relative = f'predicted_masks/{scene_id}_{inst_ndx:03d}.json'
        main_txt_lines.append(f'{mask_path_relative} {inst_sem_label} 1.0')
        # save the instance mask to a file in the predicted_masks dir
        mask_path = join(inst_predsformat_out_dir, mask_path_relative)
        write_json(mask_path, rle_encode(inst_mask))

    # save the main txt file
    with open(main_txt_file, 'w') as f:
        f.write('\n'.join(main_txt_lines))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str,
                        default='/YourScanNetPath', help='path to scannet dataset')
    parser.add_argument('--scene_id', type=str, default=None)
    parser.add_argument('--mask_name', type=str, default='semantic-sam',
                        help='which group of mask to use(fast-sam, sam-hq...)')
    parser.add_argument('--test', default=False, action='store_true',
                        help='just a case for tweak parameter, will save file in particular names')
    parser.add_argument('--text', type=str, help='save file with a prefix')
    parser.add_argument('--view_freq', type=int, default=5,
                        help='how many views to select one view from')
    parser.add_argument('--thres_connect', type=str, default="0.9,0.3,5",
                        help='dynamic threshold for progresive region growing, in the format of "start_thres,end_thres,stage_num')
    parser.add_argument('--dis_decay', type=float, default=0.5,
                        help='weight decay for calculating seg-region affinity')
    parser.add_argument('--thres_dis', type=float, default=0.15,
                        help='distance threshold for visibility test')
    parser.add_argument('--thres_merge', type=int, default=200,
                        help='thres to merge small isolated regions in the postprocess')
    parser.add_argument('--max_neighbor_distance', type=int, default=2,
                        help='max logical distance for taking priimtive neighbors into account')
    parser.add_argument('--similar_metric', type=str, default='2-norm',
                        help='metric to compute similarities betweeen primitives, see utils.py/calcu_similar() for detail')
    parser.add_argument('--thres_trunc', type=float, default=0.,
                        help="trunc similarity that is under thres to 0")
    parser.add_argument('--from_points_thres', type=float, default=0,
                        help="if > 0, use points as primitives for region growing in the first stage")
    parser.add_argument('--use_torch', action='store_true',
                        help='use torch version or numpy version')
    parser.add_argument('--scannetpp', default=False,
                        action='store_true', help='use scannet++ dataset(not debug yet)')
    parser.add_argument('--eval_dir', type=str, help='where to save eval res')

    args = parser.parse_args()

    train_split, val_split = utils.get_splits(args.base_dir, args.scannetpp)

    seg_split = val_split

    seg_split = sorted(seg_split)

    if args.scene_id is not None:
        seg_split = args.scene_id.split(',')

    thres_connects = args.thres_connect.split(',')
    assert len(thres_connects) == 3
    args.thres_connect = np.linspace(
        float(thres_connects[0]), float(thres_connects[1]), int(thres_connects[2]))
    
    for scene_id in tqdm(seg_split):

        
        if os.path.exists(os.path.join(args.eval_dir, scene_id+'.txt')):
            print("already save!")
            continue
        args.scene_id = scene_id
        print(scene_id)
        everything_seg(args)
       

