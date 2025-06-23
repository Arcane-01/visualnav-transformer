import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
import argparse
from dpt.dpt import DptTrtInference
from sdf import pcd_to_laser, create_occupancy_map, opencv_sdf
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import Image
import time
import time
from sensor_msgs.msg import PointCloud2, PointField
from parameters import *
import torch

dpt = DptTrtInference(ENGINE_PATH, 1, IMAGE_SHAPE, OUTPUT_SHAPE)
pcd_pub = rospy.Publisher(PCD_TOPIC, PointCloud2, queue_size=1)
depth_pub = rospy.Publisher(DEPTH_TOPIC, Image, queue_size=1)
tsdf_pub = rospy.Publisher(TSDF_TOPIC, Image, queue_size=1)
occ_pub = rospy.Publisher("/local_map", OccupancyGrid, queue_size=1)
cam_height = 1
bridge = CvBridge()

def coords_to_index(map, trajs, x_min=X_BOUNDS[0], y_min=Y_BOUNDS[0], resolution=RESOLUTION):
    grid_y, grid_x = map.shape
    indices = []
    for traj in trajs:
        traj_indices = []
        x_indices = ((traj[:, 0] - x_min) / resolution).astype(int)
        y_indices = ((traj[:, 1] - y_min) / resolution).astype(int)
        # Check bounds
        valid_mask = (
            (x_indices >= 0) & (x_indices < grid_x) &
            (y_indices >= 0) & (y_indices < grid_y)
        )
        for xi, yi, valid in zip(x_indices, y_indices, valid_mask):
            if valid:
                traj_indices.append((yi, xi))  # (row, col) indexing
            else:
                traj_indices.append(None)  # or skip, or use (-1, -1)
        indices.append(traj_indices)
    return indices

def cost_function(tsdf, trajs, x_min=X_BOUNDS[0], y_min=Y_BOUNDS[0], resolution=RESOLUTION):
    trajs_cost = []
    grid_y, grid_x = tsdf.shape
    for traj in trajs:
        cost = 0
        x_indices = ((traj[:, 0] - x_min) / resolution).astype(int)
        y_indices = ((traj[:, 1] - y_min) / resolution).astype(int)
        # Check bounds
        valid_mask = (
            (x_indices >= 0) & (x_indices < grid_x) &
            (y_indices >= 0) & (y_indices < grid_y)
        )
        if not np.all(valid_mask):
            print("Warning: Some waypoints are out of bounds and will be ignored.")
        cost += np.sum(tsdf[y_indices[valid_mask], x_indices[valid_mask]])
        trajs_cost.append(cost)
    return trajs_cost

def pub_ros_occupancy(grid, resolution=RESOLUTION):
    height, width = grid.shape  
    grid = grid.astype(np.int8)
    msg = OccupancyGrid()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "base_link"  # local frame

    msg.info.resolution = resolution
    msg.info.width = width
    msg.info.height = height
    # Set the origin at bottom-left corner (shifted backward so ego is at (0,0))
    origin_x = 0.0  
    origin_y = -4.0  

    msg.info.origin = Pose(
        position=Point(x=origin_x, y=origin_y, z=0),
        orientation=Quaternion(x=0, y=0, z=0, w=1)
    )

    # Convert grid to int8 and flatten
    msg.data = grid.flatten(order='C').tolist()
    occ_pub.publish(msg)

def image_callback(img):
    start_time = time.time()
    # img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = img[None]  # B, C, H, W
    dt = time.time()
    depth, pcd_pts = dpt(img)
    
    # Camera to Lidar tf
    T_lidar_camera = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )
    pcd_pts_homogeneous = np.hstack([pcd_pts, np.ones((pcd_pts.shape[0], 1))])
    pcd_pts_lidar = (T_lidar_camera @ pcd_pts_homogeneous.T).T[:, :3]

    lt = time.time()
    laser_points = pcd_to_laser(pcd_pts_lidar)
    ot = time.time()
    occupancy = create_occupancy_map(laser_points, X_BOUNDS, Y_BOUNDS, RESOLUTION)
    st = time.time()
    tsdf = opencv_sdf(occupancy)
    tt = time.time()
    print('depht/pcd, laser, occ, sdf :', (lt-dt)*1000, (ot-lt)*1000, (st-ot)*1000, (tt-st)*1000)


    depth = depth.squeeze().cpu().numpy().astype(np.uint8)
    
    color_time = time.time()
    colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_PLASMA)
    colored_tsdf = cv2.applyColorMap(tsdf.astype(np.uint8), cv2.COLORMAP_PLASMA)

    end_time = time.time()
    print('Color Time :', (end_time - color_time)*1000)
    print("Total Time : ", (end_time - start_time)*1000)

    # Publish the o3d pcd 
    pub_ros_occupancy(100*occupancy)
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "base_link"
    # Add cam_height to z coordinate
    pcd_pts_baselink = pcd_pts_lidar.copy()
    pcd_pts_baselink[:, 2] += cam_height
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]
    pc2_msg = pc2.create_cloud(header, fields, pcd_pts_baselink)
    pcd_pub.publish(pc2_msg)

    # Publish the colored depth image
    depth_msg = bridge.cv2_to_imgmsg(colored_depth, encoding="bgr8")
    depth_msg.header = header
    depth_pub.publish(depth_msg)

    # Publish the colored TSDF image
    # tsdf_msg = bridge.cv2_to_imgmsg(colored_tsdf, encoding="bgr8")
    # tsdf_msg.header = header
    # tsdf_pub.publish(tsdf_msg)
    torch.cuda.empty_cache()

    return tsdf

def main():
    rospy.init_node("DepthAnything", anonymous=False)
    print("Node Started")
    img_sub = rospy.Subscriber(IMAGE_TOPIC, Image, image_callback, queue_size=1)
    rospy.spin()
    
if __name__ == '__main__':
    main()