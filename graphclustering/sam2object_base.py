import numpy as np
import time
from tqdm import tqdm
import scipy
import open3d as o3d
import helpers.sam2object_utils as utils
import os
import cv2
from scipy.spatial import ConvexHull
n_workers = 20

def random_selection(N, n):
    return np.random.choice(np.arange(0, N), size=n, replace=False)

def evenly_spaced_selection(N, n):
    interval = (N - 1) / (n - 1)
    return np.round(np.arange(0, N, interval)[:n]).astype(int)


def calculate_mask_observation_probability(points_seen, instance_mask, ratio=0.5):

    unique_masks = np.unique(instance_mask)  
    total_masks = len(unique_masks)          
    successfully_observed_masks = 0          

    for mask_id in unique_masks:
        mask_indices = (instance_mask == mask_id).flatten()
        mask_points_seen = points_seen[mask_indices, :]  
        
        observed_points = np.any(mask_points_seen != 0, axis=1)  
        observed_ratio = np.sum(observed_points) / observed_points.shape[0]  

        if observed_ratio >= ratio:
            successfully_observed_masks += 1

    success_probability = (successfully_observed_masks / total_masks) * 100
    return success_probability


def calculate_mask_coverage(points_seen, instance_mask):

    unique_masks = np.unique(instance_mask)  
    mask_coverage = {} 

    for mask_id in unique_masks:
        mask_indices = (instance_mask == mask_id).flatten()
        mask_points_seen = points_seen[mask_indices, :]  
        
        mask_observed_per_frame = np.any(mask_points_seen != 0, axis=0)  
        
        import pdb; pdb.set_trace()
        coverage = np.sum(mask_observed_per_frame) / unique_masks.shape[0] * 100
        mask_coverage[mask_id] = coverage

    return mask_coverage

def calculate_coverage(points_seen):

    points_observed = np.any(points_seen != 0, axis=1)
    
    observed_count = np.sum(points_observed)
    
    total_points = points_seen.shape[0]
    
    coverage = (observed_count / total_points) * 100
    return coverage

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP != rootQ:
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                self.parent[rootQ] = rootP
                self.rank[rootP] += 1

def process_affinities(affinity_matrix, threshold, ratio_threshold=0.8):
    n = affinity_matrix.shape[0]
    connected = [[False] * n for _ in range(n)]
    
    # Step 1: Initial connection based on threshold
    for i in range(n):
        for j in range(n):
            if affinity_matrix[i][j] >= threshold:
                connected[i][j] = True
    
    # Step 2: Check paths of length 2
    for i in range(n):
        for j in range(n):
            if i != j and connected[i][j]:
                high_high = 0
                high_low = 0
                for k in range(n):
                    if k != i and k != j:
                        if affinity_matrix[i][k] >= threshold and affinity_matrix[k][j] >= threshold:
                            high_high += 1
                        elif affinity_matrix[i][k] >= threshold or affinity_matrix[k][j] >= threshold:
                            high_low += 1
                if high_low > 0 and high_high / high_low < ratio_threshold:
                    connected[i][j] = False
    
    # Step 3: Union-Find to merge connected superpixels
    uf = UnionFind(n)
    for i in range(n):
        for j in range(n):
            if connected[i][j]:
                uf.union(i, j)
    
    # Extract clusters
    clusters = {}
    for i in range(n):
        root = uf.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)
    
    return clusters


class SAM2OBJECTBase:
    def __init__(self, points, args):
        """
        :param points_objness: (x, y, z, objectness)
        """
        self.points = points
        self.N = points.shape[0]
        self.max_neighbor_distance = args.max_neighbor_distance
        self.similar_metric = args.similar_metric
        self.args = args
        self.view_freq = args.view_freq
        self.dis_decay = args.dis_decay
        self.key_path = args.key_path


    def query_2d_point(self, seg_id, frame_idx, color_pixes, vis_img_path, save_path):
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, str(frame_idx) + '_' + str(seg_id) + '.jpg')
        project_img_coord = color_pixes[self.seg_members[seg_id], frame_idx,:]
        vis_img_path = os.path.join(vis_img_path, str(frame_idx) + '.jpg')
        vis_img = cv2.imread(vis_img_path)
        # process vis

        # save
        for point in project_img_coord:
            cv2.circle(vis_img, tuple(point), 5, (0, 255, 0), -1)
        cv2.imwrite(save_path, vis_img)


    def get_point_cloud_distances(self, pose, frame_idx, points_world):

        """
        Computes the distance from each point in a point cloud to the camera.

        Parameters
        ----------
        pose : np.ndarray
            The 4x4 camera pose matrix.
        points_world : np.ndarray
            An (Nx3) point cloud, where each row represents a point's coordinates
            in the world coordinate system.

        Returns
        -------
        distances : np.ndarray
            An (N,) array containing the distance from each point to the camera.
        """
        pose = pose[frame_idx]
        pose_inv = np.linalg.inv(pose)
        num = points_world.shape[0]
        points_homogeneous = np.hstack((points_world, np.ones((num, 1))))  # N x 4

        points_camera_homogeneous = (pose_inv @ points_homogeneous.T).T  # N x 4

        distances = np.linalg.norm(points_camera_homogeneous[:, :3], axis=1)    
        
        return distances


    def get_all_point_cloud_distances(self, pose, points_world):
        """
        Computes the distance from each point in a point cloud to the camera.

        Parameters
        ----------
        pose : np.ndarray
            The 4x4 camera pose matrix.
        points_world : np.ndarray
            An (Nx3) point cloud, where each row represents a point's coordinates
            in the world coordinate system.

        Returns
        -------
        distances : np.ndarray
            An (N,) array containing the distance from each point to the camera.
        """
        pose_inv = np.linalg.inv(pose)
        num = points_world.shape[0]
        points_homogeneous = np.hstack((points_world, np.ones((num, 1))))  # N x 4


        points_camera_homogeneous = (pose_inv @ points_homogeneous.T).T  # N x 4

        distances = np.linalg.norm(points_camera_homogeneous[:, :3], axis=1)    
        
        return distances

    def compute_cloud_volume_and_area(self):
        seg_volumes, seg_areas = self.compute_seg_volumes_and_areas()

        total_volume = np.sum(seg_volumes)
        total_area = np.sum(seg_areas)

        return total_volume, total_area
    def compute_seg_volumes_and_areas(self):
        seg_ids = np.unique(self.seg_ids)
        seg_volumes = np.zeros(len(seg_ids))
        seg_areas = np.zeros(len(seg_ids))
        
        for seg_id in np.unique(seg_ids):
            seg_member_ids = self.seg_members[seg_id]
            seg_points = self.points[seg_member_ids]

            hull = ConvexHull(seg_points)
            seg_volume = hull.volume
            seg_volumes[seg_id] = seg_volume

            seg_triangles = hull.simplices
            seg_area = 0
            for triangle in seg_triangles:
                p1 = seg_points[triangle[0]]
                p2 = seg_points[triangle[1]]
                p3 = seg_points[triangle[2]]
                seg_area += 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
            seg_areas[seg_id] = seg_area

        return seg_volumes, seg_areas

    def assign_label(self,
                     points,
                     thres_connect,
                     vis_dis,
                     max_neighbor_distance=2,
                     similar_metric='2-norm'):
        """
        assign instance labels for all points in the scene 
        
        :param points: (N, 3), points in world coordinate
        :param thres_connect: the threshold for judging whether two superpoints are connected
        :param vis_dis: the distance threshold for judging whether a point is visible
        :param max_neighbor_distance: the max logical distance of indirect neighbors to take into account
        :param similar_metric: the metric to calculate the similarity between two primitives
        :return points_labels: (N,), resulting instance labels of all points
        """
        pre_time = time.time()
        # project N points to M images
        pts_cam, color_pixes, depth_pixes = self.parallel_world2cam_pixel(
            points, 
            self.color_intrinsics, 
            self.depth_intrinsics,
            self.poses)  # (N, M, 3), (N, M, 2)
        project_time = time.time() - pre_time
        print('parallel_world2cam_pixel:', time.time() - pre_time)

        points_label, points_seen = self.get_points_label_seen(
            pts_cam,
            color_pixes,
            depth_pixes,
            semantic=False,
            discard_unseen=True,
            vis_dis=vis_dis)  # (N, k)
        distance_all =  self.get_all_point_cloud_distances(self.poses, self.points)
        

        points_labels = None

        # use points as primitive for region growing
        if (hasattr(self.args,"from_points_thres") 
                and self.args.from_points_thres > 0):
            initial_labels = np.arange(self.N, dtype=int) + 1
            self.seg_ids, self.seg_num, self.seg_members, self.seg_direct_neighbors = self.get_seg_data(
                # base_dir=self.base_dir,
                base_dir='/Your/Path/To/Your/ScanNet',  # Change this to your data directory
                scene_id=self.scene_id,
                max_neighbor_distance=1,
                seg_ids=initial_labels,
                point_level=True)

            seg_adj = self.get_seg_dok_adjacency(
                points_label=points_label,
                points_seen=points_seen)
            points_labels = self.assign_seg_label(
                seg_adj,
                self.args.from_points_thres,
                max_neighbor_distance=1,
                dense_neighbor=True).astype(int)
        # distance_all =  self.get_all_point_cloud_distances(self.poses, self.points)

        # progressive region growing,
        # the resulting oversegmentations of last iteration can be the primitive of next iteration.
        all_stage_pre_time = time.time()
        stage_pre_time = []
        stage_time = []
        for i in range(len(thres_connect)):
            stage_tmp_time = time.time()
            stage_pre_time.append(stage_tmp_time)
            self.seg_ids, self.seg_num, self.seg_members, self.seg_indirect_neighbors = self.get_seg_data(
                base_dir=self.base_dir,
                scene_id=self.scene_id,
                max_neighbor_distance=self.max_neighbor_distance,
                seg_ids=points_labels)

            self.seg_direct_neighbors = self.seg_indirect_neighbors[0]
            seg_adj = self.get_seg_adjacency(
                points_any=points,
                distance=distance_all,
                similar_meric=similar_metric,
                points_label=points_label,
                points_seen=points_seen,
                num=i)
            
            seg_labels = self.assign_seg_label(
                seg_adj,
                thres_connect[i],
                max_neighbor_distance=max_neighbor_distance)

            # only conduct postprocessing in the last iteration
            if i == len(thres_connect) - 1 and self.args.thres_merge > 0:
                seg_labels = self.merge_small_segs(seg_labels,
                                                   self.args.thres_merge,
                                                   seg_adj)

                # assign primitive labels to member points
            points_labels = np.zeros(self.N, dtype=int)
            for jj in range(self.seg_num):
                label = seg_labels[jj]
                points_labels[self.seg_members[jj]] = label
        
        return points_labels

    def assign_seg_label(self, 
                         adj, 
                         thres_connect, 
                         max_neighbor_distance, 
                         dense_neighbor=False):
        """implement primitive level region growing
        :param adj:dense metrix or sparse.dok_matrix:(s,s)
        :return: seg_labels:(s,)
        """
        pre_time = time.time()

        assign_id = 1
        seg_labels = np.zeros(self.seg_num, dtype=np.float32)
        for i in range(self.seg_num): 
            if seg_labels[i] <= 0:
                queue = []
                queue.append(i)
                seg_labels[i] = assign_id
                group_points_count = self.seg_member_count[i]
                seg_parents = np.full([self.seg_num], -1, dtype=int)

                while queue:
                    v = queue.pop(0)
                    if dense_neighbor:
                        js = self.seg_direct_neighbors[v]
                    else:
                        js = self.seg_direct_neighbors[v].nonzero()[0]

                    seg_parents[js] = v
                    for j in js:
                        if seg_labels[j] != 0:
                            continue
                        connect = self.judge_connect( #判断是否可以和邻居连接
                            adj, v, j, thres_connect, 
                            seg_labels, assign_id, 
                            group_points_count, 
                            max_neighbor_distance, 
                            decay=self.dis_decay)
                        
                        if not connect:
                            continue
                        seg_labels[j] = assign_id
                        group_points_count += self.seg_member_count[j]
                        queue.append(j)
                assign_id += 1


        return seg_labels

    def assign_seg_label_topk(self, 
                         adj, 
                         thres_connect, 
                         max_neighbor_distance, 
                         dense_neighbor=False):
        """implement primitive level region growing
        :param adj:dense metrix or sparse.dok_matrix:(s,s)
        :return: seg_labels:(s,)
        """
        pre_time = time.time()

        assign_id = 1
        seg_labels = np.zeros(self.seg_num, dtype=np.float32)
        for i in range(self.seg_num): 
            if seg_labels[i] <= 0:
                queue = []
                queue.append(i)
                seg_labels[i] = assign_id
                group_points_count = self.seg_member_count[i]
                seg_parents = np.full([self.seg_num], -1, dtype=int)

                while queue:
                    v = queue.pop(0)
                    if dense_neighbor:
                        js = self.seg_direct_neighbors[v]
                    else:
                        js = self.seg_direct_neighbors[v].nonzero()[0]

                    seg_parents[js] = v
                    for j in js:
                        if seg_labels[j] != 0:
                            continue
                        connect = self.judge_connect( 
                            adj, v, j, thres_connect, 
                            seg_labels, assign_id, 
                            group_points_count, 
                            max_neighbor_distance, 
                            decay=self.dis_decay)
                        
                        if not connect:
                            continue
                        seg_labels[j] = assign_id
                        group_points_count += self.seg_member_count[j]
                        queue.append(j)
                assign_id += 1

        print("time for region_growing:", time.time() - pre_time)
        print("number of region:", assign_id - 1)
        return seg_labels  # (s, )

    def parallel_world2cam_pixel(self, 
                                 points, 
                                 color_intrinsics, 
                                 depth_intrinsics, 
                                 poses):
        if self.args.use_torch:
            points_cam, color_points_pixel, depth_points_pixel = utils.torch_world2cam_pixel(
                points, color_intrinsics, depth_intrinsics, poses)
        else:
            points_cam, color_points_pixel, depth_points_pixel = utils.world2cam_pixel(
                points, color_intrinsics, depth_intrinsics, poses)

        return points_cam, color_points_pixel, depth_points_pixel

    def get_points_label_seen(self, 
                              pts_cam, 
                              color_pixes, 
                              depth_pixes, 
                              semantic=False, 
                              discard_unseen=True, 
                              vis_dis=0.15):
        """get labels and seen flag of all points in all views

        :param pts_cam: (N, M, 3), transformed to camera coordinate
        :param color_pixes, depth_pixes: (N, M, 2), projected pixel locations to color and depth images
        :param semantic: use sam mask or sematic mask 
        :param discard_unseen: whether to set labels of unseen points to 0
        :param vis_dis: the distance threshold for judging whether a point is visible
        
        :return all_label: (N, M), labels of all points in all views
        :return all_seen_flag: (N, M), seen flag of all points in all views
        """

        batch_size = 50000
        all_label, all_seen_flag = np.zeros(
            [self.N, self.M], dtype=np.float32), np.zeros([self.N, self.M], dtype=bool)

        for start_id in tqdm(range(0, self.N, batch_size)):
            p_cam0 = pts_cam[start_id: start_id + batch_size]
            color_pix0 = color_pixes[start_id: start_id + batch_size]
            depth_pix0 = depth_pixes[start_id: start_id + batch_size]
            cw0, ch0 = np.split(color_pix0, 2, axis=-1)
            cw0, ch0 = cw0[..., 0], ch0[..., 0]  # (N, M)
            bounded_flag0 = (0 <= cw0)*(cw0 <= self.CW - 1)*(0 <= ch0)*(ch0 <= self.CH - 1)  # (N, M)
            """get labels from masks
            Note that results from invalid indices are meaningless. 
            However, we clip invalid indices and also query the results 
            in order to obtain a regular (N, M) array.
            When return the results, the validity must be considered.
            """
            if not semantic:
                # (N, M), querying labels from masks (M, H, W) by h (N, M) and w (N, M)
                label0 = self.masks[np.arange(self.M), 
                                    ch0.clip(0, self.CH - 1), 
                                    cw0.clip(0, self.CW - 1)]
            else:
                label0 = self.semantic_masks[np.arange(self.M), 
                                             ch0.clip(0, self.CH - 1), 
                                             cw0.clip(0, self.CW - 1)]
            dw0, dh0 = np.split(depth_pix0, 2, axis=-1)
            dw0, dh0 = dw0[..., 0], dh0[..., 0]  # (N, M)

            # judge whether the point is visible
            real_depth0 = p_cam0[..., -1]  # (N, M)
            capture_depth0 = self.depths[np.arange(self.M), 
                                         dh0.clip(0, self.DH - 1), 
                                         dw0.clip(0, self.DW - 1)]  # (N, M), querying depths
            visible_flag0 = np.isclose(
                real_depth0, capture_depth0, rtol=vis_dis)
            seen_flag = bounded_flag0 * visible_flag0

            if discard_unseen:
                label0 = label0 * seen_flag  # set label of invalid point to 0

            all_seen_flag[start_id: start_id + batch_size] = seen_flag
            all_label[start_id: start_id + batch_size] = label0

        return all_label, all_seen_flag

    # ====== about adjacency ======
    """
    The three functions below are used to calculate the adjacency when the primitive of growing is points.
    """

    def get_seg_dok_adjacency(self, points_label, points_seen):
        """need self.seg_direct_neighbors to has regular shape and be dense matrix(N,k)
        
            points_label:(N,M)
            points_seen:(N,M)
        """
        neighbors = self.seg_direct_neighbors  # (N,k)
        connect_mat, seen_mat = self.get_dense_connect_seen(
            neighbors, points_label, points_seen)  # (N,k)
        adj = self.get_dok_adjacency_from_connect_seen(
            neighbors, connect_mat=connect_mat, seen_mat=seen_mat)

        return adj
    def mesh_to_pointcloud(self,mesh):
        """
        Convert an Open3D TriangleMesh to a PointCloud.
        
        Args:
            mesh (open3d.geometry.TriangleMesh): The input mesh.
            
        Returns:
            open3d.geometry.PointCloud: The converted point cloud.
        """
        # Extract the vertices from the mesh
        vertices = np.asarray(mesh.vertices)
        
        # Create a PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        
        # Copy the vertex colors to the PointCloud
        if mesh.has_vertex_colors():
            vertex_colors = np.asarray(mesh.vertex_colors)
            pcd.colors = o3d.utility.Vector3dVector(vertex_colors)
        
        return pcd

    def get_dense_connect_seen(self, neighbors, points_label, points_seen):
        labels_allk = points_label[neighbors]  # (N,k,M)
        seen_allk = points_seen[neighbors]  # (N,k,M)

        connect_mat, seen_mat = [], []
        for i in tqdm(range(labels_allk.shape[1]), desc='get connect mat'):
            label1 = labels_allk[:, i]  # (N,M)
            seen1 = seen_allk[:, i]  # (N,M)

            f_seen = points_seen * seen1  # (N,M)
            f_connect = f_seen * (points_label == label1)  # (N, M)

            connect = f_connect.sum(axis=-1)  # (N, M) -> (N, )
            seen = f_seen.sum(axis=-1)  # (N, M), -> (N, )

            connect_mat.append(connect)
            seen_mat.append(seen)

        connect_mat = np.stack(connect_mat, axis=-1)  # (N, k)
        seen_mat = np.stack(seen_mat, axis=-1)

        return connect_mat, seen_mat

    def get_dok_adjacency_from_connect_seen(self, neighID_mat, connect_mat, seen_mat):
        """
        :param neighID_mat: (N, k)
        :param connect_mat: (N, k)
        :param seen_mat: (N, k)
        :return: adj: sparse dok matrix (N, N)
        """
        N_any, k_graph = neighID_mat.shape
        rows_ori = neighID_mat[:, 0]  # (N, )
        # Note that the operations follow the row-major order
        rows = np.repeat(rows_ori, k_graph)  # (N*k, ), (0,0,0,0,1,1,1,1,...)
        cols = neighID_mat.flatten()  # (N*k, ), (a0, a1, a2, a3, b0, b1, b2, b3, ...)
        connects = connect_mat.flatten()  # (N*k, )
        seens = seen_mat.flatten()  # (N*k, )

        adj_connect = scipy.sparse.coo_array((connects, (rows, cols)), shape=(N_any, N_any), dtype=np.float32).todok()  # (N, N)
        adj_seen = scipy.sparse.coo_array((seens, (rows, cols)), shape=(N_any, N_any), dtype=np.float32).todok()  # (N, N)
        if adj_seen.nonzero()[0].size == 0:
            return adj_seen
        # assert adj_seen.nonzero()[0].size, "Nothing seen. Probably you use the wrong gsam directory, i.e., wrong `text` and `thres_gsam`."
        adj = scipy.sparse.dok_matrix(adj_seen.shape, dtype=np.float32)
        adj[adj_seen.nonzero()] = adj_connect[adj_seen.nonzero()] / adj_seen[adj_seen.nonzero()]

        # Sometimes, j is a neighbor of i (adj[i,j] = score), but i is not a neighbor of j (adj[j, i] = 0)
        # make sure adj is a symmetric matrix, by assign a[i,j] = a[j,i] = max(a[i,j], a[j,i])
        adj = adj.tocsr()
        r, c = adj.nonzero()
        adj[r, c] = adj[c, r] = np.maximum(adj[r, c], adj[c, r])

        return adj

    """
    The three functions below are used to calculate the adjacency when the primitive of growing is superpoints.
    """

    def get_seg_adjacency(self, 
                          points_any,
                          distance, 
                          similar_meric, 
                          points_label, 
                          points_seen,
                          num):
        """
        :params similar_meric: the metric to calculate the similarity between two primitives
        :params points_label: (N, M), labels of all points in all views
        :params points_seen: (N, M), seen flag of all points in all views        
        
        :return: adjacency_mat, (s,s): adjacency between each pair of neighboring segs
        """
        pre_time = time.time()

        similar_mat, confidence_mat, multiview_similar_mat = self.get_neighbor_seg_similar_confidence_matrix(
            points_label, 
            points_seen, 
            distance,
            self.max_neighbor_distance, 
            similar_meric,
            self.args.thres_trunc,
            multi_view=True)  # (N, N)

        adjacency_mat, adj_2 = self.get_seg_adjacency_from_similar_confidence(
            similar_mat, confidence_mat, multiview_similar_mat=multiview_similar_mat)  # (N, N)

        return adj_2


    def calculate_label_ratios(self, matrix, label_num):
        matrix = matrix.astype(int)
        res = np.zeros([matrix.shape[0], label_num])
        
        for i in range(matrix.shape[0]):
            u_v, u_c = np.unique(matrix[i], return_counts=True)
            res[i][u_v] = u_c
        
        res_f = np.sum(res[:,1:], axis=0)
        total_labels = np.sum(res_f)
        if total_labels == 0:
            return res_f
        label_ratios = res_f / (matrix.shape[0]) 
 
        return label_ratios
    

    def get_neighbor_seg_similar_confidence_matrix(self, 
                                                   points_label, 
                                                   points_seen, 
                                                   distance,
                                                   max_neighbor_distance, 
                                                   similar_metric, 
                                                   thres_trunc=0., 
                                                   process_num=12,
                                                   multi_view=False
                                                   ):
        """
        
        :param points_label: (N, M), labels of all points in all views
        :param points_seen: (N, M), seen flag of all points in all views
        :param max_neighbor_distance: the max logical distance of indirect neighbors to take into account
        :param similar_metric: the metric to calculate the similarity between two primitives
        :param thres_trunc: the threshold for discarding the similarity between two primitives if their confidence is too low
        :param process_num: the number of processes to use
        :param use_torch: whether to use torch to accelerate the calculation of the similarity and confidence
        
        :return similar_sum (s,s): weight sum of similar score in every view
        :return confidence (s,s): sum of confidence of how much we can trust the similar score in every view
        """
        seg_neighbors = self.seg_indirect_neighbors[max_neighbor_distance-1]  # binary matrix, (s,s)
        seg_members = self.seg_members  # dict {seg_id: point_array}
        seg_ids = self.seg_ids
        label_num = int(points_label.max()) + 1

        # first get visible ratio of each seg in every view
        seg_seen0 = np.zeros([self.seg_num, self.M], dtype=np.float32)  # (s,m)
        seg_distance0 = np.zeros([self.seg_num, self.M], dtype=np.float32)  # (s,m)
        # label_num - 1 排除0
        seg_label_seen0 = np.zeros([self.seg_num, label_num - 1], dtype=np.float32)  # (s,lr)
        label_dis = np.arange(1, label_num)
        for seg_id, members in seg_members.items():
            p_m = points_label[members]
            seg_seen0[seg_id] = ((p_m > 0).sum(axis=0)) / members.shape[0]  # (mem,m) -sum-> (m,)   #将mask中黑色区域也视为被遮挡
            seg_label_seen0[seg_id] = self.calculate_label_ratios(p_m, label_num) #超体素在所有m帧上label分布
        
            p_distance = distance[members]
            seg_distance0[seg_id] = np.nanmean(p_distance, axis=0)

        if self.args.use_torch:
            if multi_view:
                similar_sum, confidence_sum, multiview_similar = utils.torch_get_similar_confidence_matrix_multiview_2(
                    seg_neighbors, seg_ids,
                    seg_seen0, seg_distance0, points_label,
                    similar_metric,
                    seg_label_seen0,
                    thres_trunc,
                    multi_view=multi_view,
                    key_list_path=self.key_path
                    )
            else:
                similar_sum, confidence_sum = utils.torch_get_similar_confidence_matrix_multiview_2(
                    seg_neighbors, seg_ids,
                    seg_seen0, points_label,
                    similar_metric,
                    thres_trunc,
                    multi_view=multi_view
                    )
        else:

            similar_sum, confidence_sum = utils.multiview_multiprocess_get_similar_confidence_matrix(
                seg_seen0, seg_neighbors, seg_ids,seg_distance0,
                points_label, similar_metric,
                thres_trunc, process_num)
        if multi_view:
            return similar_sum, confidence_sum, None
        else:
            return similar_sum, confidence_sum, None

    def get_seg_adjacency_from_similar_confidence(self, similar_mat, confidence_mat, multiview_similar_mat=None):
        """
        should work with the function get_neighbor_seg_similar_confidence_matrix()
        :param similar_mat: (s, s, m)
        :param confidence_mat: (s, s, m)
        :return: adj: (s, s)      
        """
        confidence_mat_one = confidence_mat[0]
        # confidence_mat_mul = confidence_mat[1]
        assert similar_mat.nonzero()[0].size > 0
        # adj = np.zeros([self.seg_num, self.seg_num])
        adj2 = np.zeros([self.seg_num, self.seg_num])
        # ori
        try:
            adj2[confidence_mat_one.nonzero()] = similar_mat[confidence_mat_one.nonzero()] / confidence_mat_one[confidence_mat_one.nonzero()]
        except:
            import pdb; pdb.set_trace()

        r, c = adj2.nonzero()
        adj2[r, c] = adj2[c, r] = np.maximum(adj2[r, c], adj2[c, r])

        return None,adj2

        # ================================

    def judge_connect(self, 
                      adj, p1_id, p2_id, 
                      thres_connect,
                      seg_labels, 
                      region_label, 
                      group_points_count, 
                      max_neighbor_distance, 
                      decay=0.5):
        """judge whether one superpoints should join the region by means of <hierarchical merging criterion>
        
        :param adj: (s, s), affinity matrix between each pair of superpoints
        :param p1_id: the id of the superpoint to be judged
        :param p2_id: the id of the superpoint to be judged
        :param thres_connect: the threshold for judging whether two superpoints are connected
        :param seg_labels: (s,), resulting labels of all superpoints so far
        :param region_label: the label of the region to join
        :param group_points_count: the number of points in the region to join
        :param max_neighbor_distance: the max logical distance of indirect neighbors to take into account
        :param decay: the decay factor for logical distance
        
        :return connect: whether the two superpoints are connected
        """
        weight_sum = 0.
        adj_sum = 0.
        weight = 1

        seg_id = p2_id #j
        for i in range(max_neighbor_distance):
            neighbor_ids = self.seg_indirect_neighbors[i][seg_id]  # (s.)
            if i > 0:
                # exclude neighbors with disance less than i
                neighbor_ids = np.logical_and(
                    neighbor_ids, 
                    np.logical_not(self.seg_indirect_neighbors[i - 1][seg_id])) #这段代码的整体逻辑是：在当前距离 i 的邻居中，排除那些在距离 i - 1 的邻居中已经存在的邻居。
            neighbor_ids = np.logical_and(neighbor_ids, 
                                          seg_labels == region_label)  # (s.) 其中就包括了v以及之前被合并的j，region_labels is assign_id 这意味着，只有在 neighbor_ids 中为真的邻居，并且其标签等于 region_label 的邻居，才会被保留。
            neighbor_ids = neighbor_ids.nonzero()[0]  # (nei,)

            # (s.)*(s.) -> (s.) -> (1.)
            adj_sum += (weight * adj[seg_id, neighbor_ids] 
                        * self.seg_member_count[neighbor_ids]).sum(0)
            weight_sum += weight * (self.seg_member_count[neighbor_ids]).sum(0)  # (s.) -> (1.)
            weight *= decay # 离的越远权重越小

        score = adj_sum / weight_sum
        return score >= thres_connect

    def merge_small_segs(self, seg_labels, merge_thres, adj):
        """postprocess segmentation results by merging small regions into neighbor regions with high affinity 

        :param seg_labels: (s,), resulting labels of all superpoints
        :param merge_thres: ithe threshold for filtering small regions
        :param adj: (s, s), affinity matrix between each pair of neighboring superpoints

        """
        seg_member_count = self.seg_member_count
        unique_labels, seg_count = np.unique(seg_labels, return_counts=True)
        region_num = unique_labels.shape[0]

        merged_labels = seg_labels.copy()
        merge_count = 0
        # 0 means the superpoint is remain to merge
        merged_mask = np.ones_like(seg_labels)
        for i in range(region_num):
            if seg_count[i] > 2:
                continue
            label = unique_labels[i]
            seg_ids = (seg_labels == label).nonzero()[0]
            if seg_member_count[seg_ids].sum() < merge_thres:
                merged_mask[seg_ids] = 0

        finished = False
        while not finished:
            flag = False  # mark whether merging happened in this iteration
            for i in range(region_num):
                label = unique_labels[i]
                seg_ids = (seg_labels == label).nonzero()[0]
                if merged_mask[seg_ids[0]] > 0:
                    continue
                seg_sims = adj[seg_ids].sum(0)
                adj_sort = np.argsort(seg_sims)[::-1]

                for i in range(adj_sort.shape[0]):
                    target_seg_id = adj_sort[i]
                    if merged_mask[target_seg_id] == 0:
                        continue  # if the target region is also too samll and has not been merged, find next target
                    if seg_sims[target_seg_id] == 0:
                        break  # no more target region can be found
                    target_label = merged_labels[target_seg_id]
                    merged_labels[seg_ids] = target_label
                    merge_count += 1
                    merged_mask[seg_ids] = 1
                    flag = True
                    break
            if not flag:
                finished = True
        # for small regions that cannot be merged, set their labels to 0
        merged_labels[merged_mask == 0] = 0

        return merged_labels

    def get_seen_image(self, points_seen):
        """print images id which can see the points

        :param points_seen: (N, M), seen flag of all points in all views
        """
        image_seen = points_seen.sum(0)
        seen_id = []
        for i in range(image_seen.shape[0]):
            if image_seen[i] > 0:
                seen_id.append(i)
        print(seen_id)
