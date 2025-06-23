# DepthAnything
ENGINE_PATH = '/home/container_user/catkin_ws/src/depth_anything_tensorrt/checkpoints/trt'
IMAGE_SHAPE = (480, 640)
OUTPUT_SHAPE = (480, 640) 
IMAGE_TOPIC = '/realsense/color/image_raw'

# Camera
# FOCAL_LENGTH = 337.208

# Occupancy Grid
X_BOUNDS = (0, 10)
Y_BOUNDS = (-4, 4)
RESOLUTION = 0.1

# Publisher Topics
PCD_TOPIC = '/pcd'
DEPTH_TOPIC = '/depth'
TSDF_TOPIC = '/tsdf'