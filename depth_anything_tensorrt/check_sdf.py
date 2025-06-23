import pickle
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def load_pickle(file):
    with open(file, "rb") as f:
        data = pickle.load(f)

    return data


def depth_to_pointcloud(depthmap, rgb_image):
    W, H = 640, 480
    fx = fy = 337.208

    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = (x - W // 2) / fx
    y = (y - H // 2) / fy
    z = np.array(depthmap)

    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    colors = rgb_image.reshape(-1, 3) / 255.0

    return points, colors


def create_occupancy_map(points, x_bounds=(0, 10), y_bounds=(-2, 2), resolution=0.1):
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    grid_x = int((x_max - x_min) / resolution)
    grid_y = int((y_max - y_min) / resolution)

    occupancy_map = np.zeros((grid_y, grid_x))

    valid_points = points[
        (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
    ]

    if len(valid_points) > 0:
        x_indices = ((valid_points[:, 0] - x_min) / resolution).astype(int)
        y_indices = ((valid_points[:, 1] - y_min) / resolution).astype(int)

        occupancy_map[y_indices, x_indices] = 1

    return occupancy_map


def create_sdf(occupancy_map, resolution=0.1):
    binary_map = (occupancy_map > 0).astype(np.uint8)
    sdf = cv2.distanceTransform(1 - binary_map, cv2.DIST_L2, 5) * resolution
    inv_sdf = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5) * resolution

    return sdf - inv_sdf


def create_occupancy_mesh(
    occupancy_map, x_bounds=(0, 10), y_bounds=(-2, 2), resolution=0.1, z_height=-0.75
):
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    occupied_indices = np.where(occupancy_map == 1)

    vertices = []
    triangles = []

    for i, j in zip(occupied_indices[0], occupied_indices[1]):
        x = x_min + j * resolution
        y = y_min + i * resolution

        v_start = len(vertices)
        vertices.extend(
            [
                [x, y, z_height],
                [x + resolution, y, z_height],
                [x + resolution, y + resolution, z_height],
                [x, y + resolution, z_height],
            ]
        )

        triangles.extend(
            [[v_start, v_start + 1, v_start + 2], [v_start, v_start + 2, v_start + 3]]
        )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color([1.0, 0.0, 0.0])

    return mesh


def visualize_maps(occupancy_map, sdf_map, x_bounds=(0, 10), y_bounds=(-2, 2)):
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.imshow(
        occupancy_map,
        cmap="binary",
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
    )
    ax1.set_title("Occupancy Map")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")

    im2 = ax2.imshow(
        sdf_map, cmap="viridis", origin="lower", extent=[x_min, x_max, y_min, y_max]
    )
    ax2.set_title("SDF Map")
    ax2.set_xlabel("X (meters)")
    ax2.set_ylabel("Y (meters)")
    plt.colorbar(im2, ax=ax2, label="Distance (meters)")

    plt.tight_layout()
    plt.show()


def main():
    # NOTE: It is recommended to directly work on the lidar frame since local planner works on that.

    # Bounds are in LiDAR frame
    x_bounds = (0, 10)  # meter
    y_bounds = (-5, 5)  # meter
    resolution = 0.05  # meter
    ground_z_bounds = (-1.0, -0.6)  # meter
    mesh_z_height = -0.8

    i = 4
    data = load_pickle(f"/home/container_user/catkin_ws/src/depth-anything-tensorrt/depthanything_pkl/{i}.pkl")

    pcd_pts, pcd_colors = depth_to_pointcloud(data["depthmap"], data["rgb"])

    T_lidar_camera = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )

    pcd_pts_homogeneous = np.hstack([pcd_pts, np.ones((pcd_pts.shape[0], 1))])
    pcd_pts_lidar = (T_lidar_camera @ pcd_pts_homogeneous.T).T[:, :3]

    all_pts = deepcopy(pcd_pts_lidar)
    all_colors = deepcopy(pcd_colors)

    ground_pts = pcd_pts_lidar[
        (pcd_pts_lidar[:, 2] >= ground_z_bounds[0])
        & (pcd_pts_lidar[:, 2] <= ground_z_bounds[1])
    ]

    occupancy_map = create_occupancy_map(ground_pts, x_bounds, y_bounds, resolution)
    sdf_map = create_sdf(occupancy_map, resolution)
    occupancy_mesh = create_occupancy_mesh(
        occupancy_map, x_bounds, y_bounds, resolution, mesh_z_height
    )

    visualize_maps(occupancy_map, sdf_map, x_bounds, y_bounds)

    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(np.asarray(ground_pts))
    ground_pcd.paint_uniform_color([0.0, 0.0, 0.0])

    pcd = o3d.geometry.PointCloud()
    coord_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(all_pts))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(all_colors))
    o3d.visualization.draw_geometries([pcd, coord_axes, ground_pcd, occupancy_mesh])


if __name__ == "__main__":
    main()