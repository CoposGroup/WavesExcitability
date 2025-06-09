import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os


def findWaveFronts(frame,
                   k=5):
    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    og_shape = frame.shape

    grad_x = cv.Sobel(frame, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(frame, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        
        
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
        
        
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    grad_flat = grad.flatten()
    best_indices = np.argsort(grad_flat)[-k:]
    points = [np.unravel_index(idx,og_shape) for idx in best_indices]
    points = np.asarray(points)[:,[1, 0]]
    return points[:, np.newaxis, :]

def findWavePeaks(frame,
                   k=5):
    og_shape = frame.shape
    frame_flat = frame.flatten()
    best_indices = np.argsort(frame_flat)[-k:]
    points = [np.unravel_index(idx,og_shape) for idx in best_indices]
    points = np.asarray(points)[:,[1, 0]]
    return points[:, np.newaxis, :]

def transform_to_int(array, min, max):
    res = np.asarray((array - min)/max * 255, dtype=np.uint8)
    return res

def save_vid(img_frames, 
            output_path =f'output.mp4',
            fps = 10,
            frame_size = (300,300),
            is_color = False):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_writer = cv.VideoWriter(output_path, fourcc, fps, frame_size, is_color)

    for frame in img_frames:
        video_writer.write(frame)

    video_writer.release() 

if __name__=="__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--wave_type", default='travelling')
    parser.add_argument("--win_size", default=5, type=int)

    args = parser.parse_args()

    wave_type = args.wave_type
    win_size = args.win_size
    dir_path = f"/home/theniche/School/REU25/WavesExcitability/Code/frames_{wave_type}"
    
    #Load frames
    from directory_tools import *
    vid_frames = getFramesInOrder(dir_path)

    #Truncate initial frames
    vid_frames = vid_frames[20:]
    vid_frames = np.asarray(vid_frames)

    #Change concentrations to image 
    max_conc = np.max(vid_frames)
    min_conc = np.min(vid_frames)
    img_frames = transform_to_int(vid_frames, min_conc, max_conc)

    #Save and load video (mainly to maintain compatibility with opencv)
    save_vid(img_frames, output_path=f'output_{wave_type}.mp4', frame_size = vid_frames[0].shape)
    vid = cv.VideoCapture(f"/home/theniche/School/REU25/WavesExcitability/Code/output_{wave_type}.mp4")


    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (win_size, win_size),
                    maxLevel = 1,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = vid.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0= findWaveFronts(old_gray, k=10).astype(np.float32)


    # Create a mask image for drawing purposes
    new_frames = []
    final_points = {(point[0],point[1]) : (point[0], point[1]) for point in p0[:,0, :]}
    final_times = {(point[0],point[1]): 0 for point in p0[:,0, :]}
    mask = np.zeros_like(old_frame)
    frame_num = 0
    while(1):
        frame_num = frame_num + 1
        ret, frame = vid.read()
        if not ret:
            print('No frames grabbed!')
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        # draw the tracksstanding
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            for key, value in final_points.items():
                if value == (c, d):
                    final_points[key] = (a,b)
                    final_times[key] = frame_num
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)
        new_frames.append(img)
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    lengths = []
    velocities = []
    for key, value in final_points.items():
        dist = np.sqrt((value[1] - key[1])**2 + (value[0] - key[0])**2)
        time = final_times[key]
        lengths.append(dist)
        if time != 0:
            velocities.append(dist/time)
        else:
            velocities.append(0.0)
    print(f'Mean Length = {np.mean(lengths)}')
    print(f'Mean Velocity = {np.median(velocities)}')

    save_vid([old_frame] + new_frames, output_path=f'optical_flow_{wave_type}.mp4', fps=3, frame_size=img_frames[0].shape, is_color=True)
    np.savetxt(f'{wave_type}_velocities.txt', velocities)
    np.savetxt(f'{wave_type}_lengths.txt', lengths)


