import os
import re
import numpy as np

def getFramesInOrder(directory):
    video_frames = []
# Helper function to extract number from filename
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')

    # Get and sort the filenames numerically
    files = sorted(os.listdir(directory), key=extract_number)

    # Loop over sorted files
    for filename in files:
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            video_frames.append(np.load(filepath))
    return video_frames