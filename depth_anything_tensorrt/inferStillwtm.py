import argparse
import os
import cv2
from PIL import Image
import numpy as np
from dpt.dpt import DptTrtInference
import time

def load_image(filepath):
    img = Image.open(filepath)  # H, W, C
    img = np.transpose(img, (2, 0, 1))  # C, H, W
    img = img[None]  # B, C, H, W
    return img.astype(np.uint8)

def run(args):
    os.makedirs(args.outdir, exist_ok=True)

    sample_img = load_image(os.path.join(args.img_dir, f"{args.start}.jpg"))
    dpt = DptTrtInference(args.engine, 1, sample_img.shape[2:], (480, 640))


    total_time = 0
    processed = 0
    
    for i in range(args.start, args.end + 1):
        img_path = os.path.join(args.img_dir, f"{i}.jpg")   
        # if not os.path.exists(img_path):
        #     print(f"Skipping {i}.jpg - file not found")
        #     continue
            
        # print(f"Processing {i}.jpg...")
        input_img = load_image(img_path)
        
        start_time = time.time()
        depth, pcd = dpt(input_img)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        print(f"  Inference time: {inference_time:.4f} seconds")
        
        # # Save depth map
        # output_path = f'{args.outdir}/{i}_depth.png'
        # depth = depth.squeeze().cpu().numpy().astype(np.uint8)
        
        # if args.grayscale:
        #     cv2.imwrite(output_path, depth)
        # else:
        #     colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        #     cv2.imwrite(output_path, colored_depth)
        
        processed += 1
    
    avg_time = total_time / processed if processed > 0 else 0
    print(f"\nProcessed {processed} images")
    print(f"Average inference time: {avg_time:.4f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run depth estimation on a range of images.')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--start', type=int, required=True, help='Start image number')
    parser.add_argument('--end', type=int, required=True, help='End image number')
    parser.add_argument('--outdir', type=str, default='./assets', help='Output directory')
    parser.add_argument('--engine', type=str, required=True, help='Path to TensorRT engine')
    parser.add_argument('--grayscale', action='store_true', help='Save in grayscale')
    args = parser.parse_args()

    run(args)