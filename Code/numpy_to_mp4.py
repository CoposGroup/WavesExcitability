import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import os

def npy_to_mp4(input_file: str, fps: int):
    # Load 3D array from .npy file
    array3d = np.load(input_file)
    
    # Check shape
    if array3d.ndim != 3:
        raise ValueError("Input array must be 3D (frames, height, width)")

    # Normalize to [0, 1]
    array3d_norm = (array3d - array3d.min()) / (array3d.max() - array3d.min())

    # Setup colormap
    cmap = cm.get_cmap('cividis')

    # Get video dimensions
    num_frames, height, width = array3d.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create output file name
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{base_name}.mp4"

    # Set up writer
    writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for frame in array3d_norm:
        colored = (cmap(frame)[:, :, :3] * 255).astype(np.uint8)
        bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    print(f"Saved video to '{output_file}'")

# Example usage:
# npy_to_mp4_with_cividis("my_data.npy", fps=15)
