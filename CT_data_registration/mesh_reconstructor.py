import numpy as np
import pathlib
import importlib
import matplotlib.pyplot as plt
# RECON_PATH = pathlib.Path("../data/20241011_phantom_mono/run2/prostate/reconstruction/droid")
RECON_PATH = pathlib.Path("/home/fj/Projects/ARPA-H/cur_vslam_models/DROID-SLAM/reconstructions/abandonedfactory")
import glob
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


class CameraPose:
    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return (
            "Metadata : "
            + " ".join(map(str, self.metadata))
            + "\n"
            + "Pose : "
            + "\n"
            + np.array_str(self.pose)
        )


def read_trajectory(filename):
    traj = []
    with open(filename, "r", encoding="utf-8") as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=" \t")
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj


volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=10.0 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
)

intrinsics = np.load(RECON_PATH/"intrinsics.npy")

fx = intrinsics[0][0]
fy = intrinsics[0][1]
cx = intrinsics[0][2]
cy = intrinsics[0][3]

color_images = np.load(RECON_PATH/"images.npy")
depth_images = 1/np.load(RECON_PATH/"disps.npy")
camera_poses =  np.load(RECON_PATH/"poses.npy") # read_trajectory(RECON_PATH/"log.txt")
camera_poses_mat = []
for i in range(camera_poses.shape[0]):
    cur_rot = R.from_quat(camera_poses[i, 3:])
    cur_pose_mat = np.eye(4)
    cur_pose_mat[:3, :3] = cur_rot.as_matrix()
    cur_pose_mat[:3, 3] = camera_poses[i, :3]

    camera_poses_mat.append(cur_pose_mat)
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    # width=584, height=328, fx=fx, fy=fy, cx=cx, cy=cy
    width=512, height=384, fx=fx, fy=fy, cx=cx, cy=cy
)

for i in range(len(color_images)):
    print(f"Integrate {i}-th image into the volume.")
    color = color_images[i]
    color = np.ascontiguousarray(color.transpose(1, 2, 0))
    color = o3d.geometry.Image((color).astype(np.uint8))
    depth = depth_images[i]
    depth = np.ascontiguousarray(depth.transpose(0, 1))
    depth = o3d.geometry.Image((depth * 255))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False
    )

    volume.integrate(rgbd, camera_intrinsic, np.linalg.inv(camera_poses_mat[i]))

print("Extract a triangle mesh from the volume and visualize it.")
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
