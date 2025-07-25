import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from geometry_msgs.msg import PoseStamped, Point, Twist, Vector3, Point
import matplotlib.pyplot as plt
import yaml
from visualization_msgs.msg import Marker, MarkerArray

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from utils import msg_to_pil, to_numpy, transform_images, load_model

from vint_train.training.train_utils import get_action
import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time
import os
from datetime import datetime

import cv2
from cv_bridge import CvBridge

# UTILS
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)


# CONSTANTS
TOPOMAP_IMAGES_DIR = "../topomaps/images"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 

# GLOBALS
context_queue = []
context_size = None  
subgoal = []
linestrip_pub = None
# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


image_counter = 0
save_images = False  
save_dir = "/home/ims/NOMAD/qualitative_results/"  

def point_proj(traj):
    cv_bridge = CvBridge()
    # Camera parameters for RealSense D455 at 640x480 resolution
    camera_matrix = np.array([
    [337.208, 0.0,    320.5],
    [0.0,    337.208, 240.5],
    [0.0,    0.0,    1.0]
    ], dtype=np.float32)

    # Distortion coefficients (5x1)
    dist_coeffs = np.array([1e-08, 1e-08, 1e-08, 1e-08, 1e-08])

    # Define the rotation and translation vectors
    R_left = np.array([
        [0.999984, -0.00329689, 0.00449711],
        [0.00329771, 0.999995, -0.000175344],
        [-0.00449651, 0.000190172, 0.99999]
    ], dtype=np.float32)

    tvec = (0, 0, 0)
    rvec, _ = cv2.Rodrigues(R_left)
    
    cv_image = cv_bridge.imgmsg_to_cv2(proj_msg, desired_encoding='rgb8')
    
    # Remove the outer loop since you now have a single trajectory
    colour = (31, 89, 181)
    points_3d = np.array([[-traj[j][1], 1, traj[j][0]] for j in range(len(traj))], dtype=np.float32)
    points_2d, _ = cv2.projectPoints(points_3d,
                                rvec, tvec,
                                camera_matrix,
                                dist_coeffs)
    points_2d = points_2d.reshape(-1, 2).astype(int)

    # Draw connecting lines
    for j in range(len(points_2d) - 1):
        start = tuple(map(int, points_2d[j]))
        end = tuple(map(int, points_2d[j+1]))
        cv2.line(cv_image, start, end, colour, 2)

    img_waypoint_msg = cv_bridge.cv2_to_imgmsg(cv_image, encoding='rgb8')
    img_waypoint_msg.header = proj_msg.header
    img_waypoint_pub.publish(img_waypoint_msg)
def _create_primitive_marker(idx, data, namespace, rgb, line_width):

    marker = Marker()
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale = Vector3(line_width, 0.01, 0)         # only scale.x used for line strip
    marker.color.r = rgb[0]                        # color 
    marker.color.g = rgb[1]                        
    marker.color.b = rgb[2]                         
    marker.color.a = 1.0                           # alpha - transparency parameter

    marker.ns = namespace
    marker.pose.orientation.w = 1.0

    for i in range(data.shape[0]):
        point = Point()
        point.x = data[i, 0]
        point.y = data[i, 1]
        point.z = 0.0
        marker.points.append(point)

    marker.id = idx
    marker.header.stamp = rospy.get_rostime()
    marker.lifetime = rospy.Duration(0.25)
    # marker.lifetime = rospy.Duration(0.0667)
    marker.header.frame_id = "base_link"

    return marker

def _visualize_trajectories(traj_optimal):
    start_point = np.array([[0, 0]])
    traj_with_start = np.concatenate([start_point, traj_optimal], axis=0)
    result = traj_with_start[:5]

    global linestrip_pub
    if linestrip_pub is not None:
        trajoptimal_marker = _create_primitive_marker(0, result, "traj_optimal", [0.12, 0.35, 0.71], 0.1)
        linestrip_pub.publish(trajoptimal_marker)

def callback_obs(msg):
    global image_counter, save_images, save_dir

    global proj_msg
    proj_msg = msg

    obs_img = msg_to_pil(msg)

    if save_images:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = f"nomad_image_{image_counter:06d}.png"
        
        filepath = os.path.join(save_dir, filename)
        obs_img.save(filepath)
        print(f"Saved image: {filepath}")
        
        # image_counter += 1

    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)


def main(args: argparse.Namespace):
    global context_size, linestrip_pub, img_waypoint_pub

     # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # load model weights
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()

    
     # load topomap
    topomap_filenames = sorted(os.listdir(os.path.join(
        TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))

    closest_node = 0
    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
    if args.goal_node == -1:
        goal_node = len(topomap) - 1
    else:
        goal_node = args.goal_node
    reached_goal = False

     # ROS
    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    image_curr_msg = rospy.Subscriber(
        IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(
        WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)

    img_waypoint_pub = rospy.Publisher(
        "/image_waypoint", Image, queue_size=1) 
    
    linestrip_pub = rospy.Publisher('/trajectory_optimal', Marker, queue_size=10)

    print("Registered with master node. Waiting for image observations...")

    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    # navigation loop
    while not rospy.is_shutdown():
        # EXPLORATION MODE
        chosen_waypoint = np.zeros(4)
        if len(context_queue) > model_params["context_size"]:
            if model_params["model_type"] == "nomad":
                obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
                obs_images = torch.split(obs_images, 3, dim=1)
                obs_images = torch.cat(obs_images, dim=1) 
                obs_images = obs_images.to(device)
                mask = torch.zeros(1).long().to(device)  

                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                goal_image = [transform_images(g_img, model_params["image_size"], center_crop=False).to(device) for g_img in topomap[start:end + 1]]
                goal_image = torch.concat(goal_image, dim=0)

                obsgoal_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
                dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                dists = to_numpy(dists.flatten())
                min_idx = np.argmin(dists)
                closest_node = min_idx + start
                print("closest node:", closest_node)
                sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

                # infer action
                with torch.no_grad():
                    # encoder vision features
                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(args.num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
                    
                    # initialize action from Gaussian noise
                    noisy_action = torch.randn(
                        (args.num_samples, model_params["len_traj_pred"], 2), device=device)
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    start_time = time.time()
                    for k in noise_scheduler.timesteps[:]:
                        # predict noise
                        noise_pred = model(
                            'noise_pred_net',
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )
                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample
                    # print("time elapsed:", time.time() - start_time)

                naction = to_numpy(get_action(naction)) # (8, 8, 2)
                sampled_actions_msg = Float32MultiArray()
                sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))
                print("published sampled actions")
                sampled_actions_pub.publish(sampled_actions_msg)
                naction = naction[0] # (8, 2)
                # print(naction)
                point_proj(naction[:5])
                _visualize_trajectories(naction)
                chosen_waypoint = naction[args.waypoint]
            else:
                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                distances = []
                waypoints = []
                batch_obs_imgs = []
                batch_goal_data = []
                for i, sg_img in enumerate(topomap[start: end + 1]):
                    transf_obs_img = transform_images(context_queue, model_params["image_size"])
                    goal_data = transform_images(sg_img, model_params["image_size"])
                    batch_obs_imgs.append(transf_obs_img)
                    batch_goal_data.append(goal_data)
                    
                # predict distances and waypoints
                batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
                batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

                distances, waypoints = model(batch_obs_imgs, batch_goal_data)
                distances = to_numpy(distances)
                waypoints = to_numpy(waypoints)
                # look for closest node
                min_dist_idx = np.argmin(distances)
                # chose subgoal and output waypoints
                if distances[min_dist_idx] > args.close_threshold:
                    chosen_waypoint = waypoints[min_dist_idx][args.waypoint]
                    closest_node = start + min_dist_idx
                else:
                    chosen_waypoint = waypoints[min(
                        min_dist_idx + 1, len(waypoints) - 1)][args.waypoint]
                    closest_node = min(start + min_dist_idx + 1, goal_node)
                print("published waypoint")
        # RECOVERY MODE
        if model_params["normalize"]:
            chosen_waypoint[:2] *= (MAX_V / RATE)  
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint
        waypoint_pub.publish(waypoint_msg)
        reached_goal = closest_node == goal_node
        goal_pub.publish(reached_goal)
        if reached_goal:
            print("Reached goal! Stopping...")
        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)


