import open3d as o3d
import copy
import numpy as np

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
    
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, ply_paths=None):
    print(":: Load two point clouds and disturb initial pose.")

    if ply_paths is None:
        demo_icp_pcds = o3d.data.DemoICPPointClouds()
        source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
        target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
    else:
        source = o3d.io.read_point_cloud(ply_paths[0])
        target = o3d.io.read_point_cloud(ply_paths[1])
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                            #  [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    trans_init = np.eye(4)
    source.transform(trans_init)
    draw_registration_result(source, target,  np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])) # np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1000
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        # Ok, I fixed it in the end. I'd blame the documentation in this case as it says: 
        # "max_correspondence_distance (float) – Maximum correspondence points-pair distance." 
        # In their tutorial this parameter is called "threshold" and I would expect the algorithm to run
        #  until the error is less than the threshold. However, the algorithm doesn't start if the error 
        # is bigger than the threshold. This should be formulated more precisely in the documentation. 
        # And especially in the tutorial this has to be fixed. If I use an threshold of 50 it works as expected.
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        5, [ # It is ransac_n, number of correspondences used in each iteration 
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 5 # 2 
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        # source, target, distance_threshold, result_ransac.transformation,
        source, target, distance_threshold, np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]),
        # o3d.pipelines.registration.TransformationEstimationPointToPlane())
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True))
    return result

voxel_size = 1  # means 0.3 mm for this dataset
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
    voxel_size,
    ply_paths=["/media/fj/Data/Projects/ARPA-H/data/20241011/CT_segmentations/run2_vslam_map_centered.ply", "/media/fj/Data/Projects/ARPA-H/data/20241011/CT_segmentations/urethra_tube_1.ply"]
    )

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
print(result_ransac)
print(result_ransac.transformation)
draw_registration_result(source_down, target_down, result_ransac.transformation)

threshold = 0.02
trans_init = np.eye(4)
print("\n\n")

print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, result_ransac.transformation)
print(evaluation)

radius_normal = voxel_size * 2 
source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                 voxel_size)
print(result_icp)
print(result_icp.transformation)
print(np.linalg.det(result_icp.transformation[:3, :3]))
draw_registration_result(source, target, result_icp.transformation)
