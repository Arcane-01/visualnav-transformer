import argparse
import os
import cv2
from PIL import Image
import numpy as np
from dpt.dpt import DptTrtInference
import time
from sdf import pcd_to_laser, occupancy_grid, opencv_sdf
import open3d as o3d

def load_image(filepath):
    img = Image.open(filepath)  # H, W, C
    img = np.transpose(img, (2, 0, 1))  # C, H, W
    img = img[None]  # B, C, H, W
    return img.astype(np.uint8)

def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    input_img = load_image(args.img)

    dpt = DptTrtInference(args.engine, 1, input_img.shape[2:], (480, 640))
    depth, pcd = dpt(input_img)
    
    laser_points = pcd_to_laser(pcd)
    occupancy = occupancy_grid(laser_points)
    tsdf = opencv_sdf(occupancy)
    
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd))
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    # o3d.io.write_point_cloud("check.ply", o3d_pcd)
    o3d.visualization.draw_geometries([o3d_pcd, axis], window_name="PointCloud Viewer")
    depth = depth.squeeze().cpu().numpy().astype(np.uint8)
    
    colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_PLASMA)
    colored_tsdf = cv2.applyColorMap(tsdf.astype(np.uint8), cv2.COLORMAP_PLASMA)
    # np.save('/home/container_user/catkin_ws/src/depth-anything-tensorrt/data/depth/1.npy', depth)
    # np.save('/home/container_user/catkin_ws/src/depth-anything-tensorrt/data/pcd/4.npy', pcd)
    # cv2.imwrite('/home/container_user/catkin_ws/src/depth-anything-tensorrt/data/rgb/4.png', cv2.imread(args.img))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run depth estimation with a TensorRT engine.')
    parser.add_argument('--img', type=str, required=True, help='Path to the input image')
    parser.add_argument('--outdir', type=str, default='./assets', help='Output directory for the depth map')
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine')
    parser.add_argument('--grayscale', action='store_true', help='Save the depth map in grayscale')
    args = parser.parse_args()

    run(args)