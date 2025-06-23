import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
import argparse
from dpt.dpt import DptTrtInference
from sdf import pcd_to_laser, create_occupancy_map, opencv_sdf
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
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

bridge = CvBridge()

def image_callback(msg):
    start_time = time.time()
    img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    header = msg.header
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]
    pc2_msg = pc2.create_cloud(header, fields, pcd_pts_lidar)
    pcd_pub.publish(pc2_msg)

    # Publish the colored depth image
    depth_msg = bridge.cv2_to_imgmsg(colored_depth, encoding="bgr8")
    depth_msg.header = msg.header
    depth_pub.publish(depth_msg)

    # Publish the colored TSDF image
    tsdf_msg = bridge.cv2_to_imgmsg(colored_tsdf, encoding="bgr8")
    tsdf_msg.header = msg.header
    tsdf_pub.publish(tsdf_msg)
    torch.cuda.empty_cache()

def main():
    rospy.init_node("DepthAnything", anonymous=False)
    print("Node Started")
    img_sub = rospy.Subscriber(IMAGE_TOPIC, Image, image_callback, queue_size=1)
    rospy.spin()
    
if __name__ == '__main__':
    main()