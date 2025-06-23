## Ros Integration with pcd and sdf:

  * Parameters are defined in ```parameters.py```
  * have rviz config saved in ```depth-anything.rviz```
  * occupancy and sdf creation functions are defined in ```sdf.py```
  * can use both ```fp16``` and ```fp32``` for engine, not much difference
  * works for WxH : 480,640, havent checked for others
  * Checkpoints can be downloaded from [here](https://iitkgpacin-my.sharepoint.com/:f:/g/personal/theyanesher_kgpian_iitkgp_ac_in/Eu0NMn5JZL5IswT32xpreMgBvVAFHQBV_YUDkYDDatHakg?e=CtirFc)
    
Pipeline Run Command:
```bash
python3 ros_pipeline.py
```

## Depth-Anything-V2 TensorRT Python

Use TensorRT to accelerate the [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) model for monocular depth estimation.

## Installation

Clone this repository and submodules:
```bash
git clone https://github.com/Stillwtm/depth-anything-tensorrt.git
git submodule init && git submodule update
```

Install dependencies:
```bash
pip install tensorrt==10.2.0.post1
```

## Model Preparation

### Download Checkpoints

Download the Depth-Anything-V2 checkpoints from [official repository](https://github.com/DepthAnything/Depth-Anything-V2), and put them under the `checkpoints` folder.

### Modify the Model

Replace the `third_party/depth_anything_v2/depth_anything_v2/dpt.py` file with the `tools/dpt.py`. In `tools/dpt.py`, we remove the `squeeze` operation in the `forward` function, which will affect the inference performance of TensorRT models.

### Convert to ONNX

```bash
python tools/export_onnx.py --checkpoint <path to checkpoint> --onnx <path to save onnx model> --input_size <dpt input size> --encoder <dpt encoder> --batch <batch size> [--dynamic_batch] [--metric] [--max_depth <max depth>]
```

### Convert ONNX to TensorRT
  * Using TensorRT 10+ which supports (execute_async_v3)
```bash
python tools/onnx2trt.py --onnx <path to onnx model> --engine <path to save trt engine> [--fp16]
```

You can also enable dynamic batch size for TensorRT engine (If you want to use dynamic batch size here, also remember to enable it in the previous ONNX model conversion step):

```bash
python onnx2trt.py --onnx <path to onnx model> --engine <path to save trt engine> [--fp16] --dynamic_batch --min_batch <minimum batch size> --max_batch <maximum batch size> --opt_batch <optimum batch size>
```

Try to decrease `max_batch` if you encounter a failure (possibly due to OOM error).

After converting the model to TensorRT, you can use the `engine` file for inference.

## Inference

For a single image, use:

```bash
python infer.py --img <path to image> --engine <path to trt engine> [--grayscale]
```

