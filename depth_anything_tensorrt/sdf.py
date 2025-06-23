import numpy as np
import cv2
# from scipy import ndimage
import matplotlib.pyplot as plt  
import numpy as np
import time
import cupy as cp
# from cucim.core.operations import morphology
import open3d as o3d

global x_min, x_max, y_min, y_max, resolution

def scipy_sdf(grid):
    start = time.time()
    inside = ndimage.distance_transform_edt(grid)
    outside = ndimage.distance_transform_edt(1 - grid)
    sdf = inside - outside  # Signed Distance Function
    end1 = time.time()
    print(f"Time taken for distance transform: {(end1 - start) * 1000} ms")
    truncation = 50.0  # meters
    tsdf = np.clip(sdf, -truncation, truncation)
    end = time.time()
    print(f"Time taken: {(end - start) * 1000} ms")
    # plt.figure(figsize=(8, 6))
    # plt.imshow(tsdf, cmap='RdYlGn')  
    # plt.colorbar(label='Signed Distance')
    # plt.show()
    
    return tsdf

def cupy_sdf():
    map = cv2.imread('/home/container_user/catkin_ws/src/rrc_lab.pgm', cv2.IMREAD_GRAYSCALE)
    map = map[200:, 500:1030]  # Crop the map to a smaller region
    print(map.shape)
    grid = cp.array(map)
    grid = grid.astype(cp.float32)/255
    start = time.time()
    inside = morphology.distance_transform_edt(grid)
    outside = morphology.distance_transform_edt(1 - grid)
    sdf_map = inside - outside  # Signed Distance Function
    end1 = time.time()
    print(f"Time taken for distance transform: {(end1 - start) * 1000} ms")
    truncation = 50.0  # meters
    tsdf_map = cp.clip(sdf_map, -truncation, truncation)
    end = time.time()
    print(f"Time taken: {(end - start) * 1000} ms")
    plt.figure(figsize=(8, 6))
    plt.imshow(cp.asnumpy(tsdf_map), cmap='RdYlGn')  
    plt.colorbar(label='Signed Distance')
    plt.show()
    
def opencv_sdf(occ_map):
    binary_uint8 = (occ_map).astype(np.uint8)
    inside_dist = cv2.distanceTransform(1 - binary_uint8, cv2.DIST_L2, 5)
    outside_dist = cv2.distanceTransform(binary_uint8, cv2.DIST_L2, 5)
    sdf = inside_dist - outside_dist 
    truncation = 30.0  # pixels
    tsdf_map = cp.clip(sdf, -truncation, truncation)
    
    # plt.figure(figsize=(15, 8))
    # plt.subplot(1,2,1)
    # plt.imshow(occ_map, cmap='gray') #, extent=[ 0, occ_map.shape[1], -occ_map.shape[0]//2, occ_map.shape[0]//2])
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Occupancu Grid')
    # plt.subplot(1,2,2)
    # plt.imshow(cp.asnumpy(tsdf_map), cmap='plasma')  
    # plt.colorbar(label='Signed Distance')
    # plt.title("SDF")
    # plt.show()

    return tsdf_map

# def occupancy_grid(points, resolution=0.1, grid_size=9):
#     grid_shape = int(grid_size/resolution), int(grid_size/resolution)
#     grid = np.zeros(grid_shape, dtype=bool)
#     points = np.array(points)
#     indices = ((points) / resolution).astype(int)
#     indices[:,1] += int(((grid_size/resolution) + 1)/2) # define the origin in middle of first row
#     for i in indices:
#         if i[1] < grid_shape[1] and i[0] < grid_shape[0]:
#             grid[i[0], i[1]] = 1
#     grid = np.flipud(np.transpose(1 - grid))  # Flip x and -y axes and invert to original occupancy
#     # Rotate the grid 90 degrees to the left (counter-clockwise)
#     # grid = np.rot90(grid)
#     # plt.figure(figsize=(8, 6))
#     # plt.imshow(1-grid, cmap='gray')
#     # plt.title('Occupancy Grid')
#     # plt.xlabel('Y')
#     # plt.ylabel('X')

#     return grid   # Return the proper occupancy grid

def create_occupancy_map(points, x_bounds=(0, 10), y_bounds=(-4, 4), resolution=0.1):
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


def depth_to_pcd(depth_map):    

    intrinsics = o3d.camera.PinholeCameraIntrinsic(width=640, height=480,
    fx=337.208, fy=337.208, cx=320.5, cy=240.5)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_map, intrinsics, depth_trunc=5)
    
    pcd.transform([[1, 0, 0, 0],
                [0,-1, 0, 0],
                [0, 0,-1, 0],
                [0, 0, 0, 1]])

    # Add coordinate frame to the visualization
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd, axis])

    return pcd

def pcd_to_laser(pcd_points):
    pcd_points = np.asarray(pcd_points)
    filtered = pcd_points[(pcd_points[:, 2] > 0.6) & (pcd_points[:, 2] < 0.9)]
    # Convert to Lidar coordinate frame
    laser_points = np.stack([filtered[:, 0], filtered[:, 1]], axis=1)

    # plt.figure(figsize=(8, 6))
    # plt.scatter(laser_points[:, 0], laser_points[:, 1], c=-laser_points[:, 0], cmap='plasma', s=2)
    # plt.colorbar(label='Y value')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Laser Points (X vs Y)')
    # print("Laser Points Shape : ", laser_points.shape)

    return laser_points
    
# def cost_function(tsdf, trajs):
#     global x_min, y_min, resolution
#     trajs_cost = []
#     grid_y, grid_x = tsdf.shape
#     for traj in trajs:
#         cost = 0
#         x_indices = ((traj[:, 0] - x_min) / resolution).astype(int)
#         y_indices = ((traj[:, 1] - y_min) / resolution).astype(int)
#         # Check bounds
#         valid_mask = (
#             (x_indices >= 0) & (x_indices < grid_x) &
#             (y_indices >= 0) & (y_indices < grid_y)
#         )
#         if not np.all(valid_mask):
#             print("Warning: Some waypoints are out of bounds and will be ignored.")
#         cost += np.sum(tsdf[y_indices[valid_mask], x_indices[valid_mask]])
#         trajs_cost.append(cost)
#     return trajs_cost

if __name__ == '__main__':
    pcd_pts = np.load('/home/container_user/catkin_ws/src/depth-anything-tensorrt/data/pcd/3.npy')
    # grid = depth_to_pcd()
    # print(grid.shape)
    T_lidar_camera = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )

    pcd_pts_homogeneous = np.hstack([pcd_pts, np.ones((pcd_pts.shape[0], 1))])
    pcd_pts_lidar = (T_lidar_camera @ pcd_pts_homogeneous.T).T[:, :3]

    pcd_to_laser = pcd_to_laser(pcd_pts_lidar)
    grid = create_occupancy_map(pcd_to_laser)
    opencv_sdf(grid)
