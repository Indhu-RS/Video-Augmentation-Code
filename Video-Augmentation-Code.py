import os
import cv2
import random
import numpy as np
import shutil
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

class Downsample(object):
    def __init__(self, ratio=1.0):
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError('ratio should be in [0.0 <= ratio <= 1.0]. ' +
                             'Please use upsampling for ratio > 1.0')
        self.ratio = ratio

    def __call__(self, clip):
        nb_return_frame = np.floor(self.ratio * len(clip)).astype(int)
        return_ind = [int(i) for i in np.linspace(1, len(clip), num=nb_return_frame)]
        return [clip[i-1] for i in return_ind]

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, clip):
        crop_h, crop_w = self.size
        frames = []
        for frame in clip:
            im_h, im_w, _ = frame.shape
            if crop_w > im_w or crop_h > im_h:
                raise ValueError('Crop size should be smaller than the image size.')

            w1 = (im_w - crop_w) // 2
            h1 = (im_h - crop_h) // 2
            frames.append(frame[h1:h1 + crop_h, w1:w1 + crop_w])
        return frames

def augment_video(input_path, output_path, crop_size, augmentation):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if augmentation == 'center_crop':
        crop_h, crop_w = crop_size
        out = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))
    else:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    if augmentation == 'horizontal':
        frames = [cv2.flip(frame, 1) for frame in frames]
    elif augmentation == 'center_crop':
        center_crop = CenterCrop(crop_size)
        frames = center_crop(frames)
    elif augmentation == 'rotated':
        angle = random.randint(-10, 10)
        M = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
        frames = [cv2.warpAffine(frame, M, (width, height)) for frame in frames]
    elif augmentation == 'downsampled':
        downsample_video = Downsample(ratio=0.5)
        frames = downsample_video(frames)
    elif augmentation == 'translated':
        dx, dy = random.randint(-50, 50), random.randint(-50, 50)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        frames = [cv2.warpAffine(frame, M, (width, height)) for frame in frames]
    elif augmentation == 'brightness':
        factor = random.uniform(0.5, 1.5)
        frames = [cv2.convertScaleAbs(frame, alpha=factor, beta=0) for frame in frames]
    elif augmentation == 'contrast':
        factor = random.uniform(0.5, 1.5)
        frames = [cv2.convertScaleAbs(frame, alpha=factor, beta=0) for frame in frames]

    for frame in frames:
        out.write(frame)

    cap.release()
    out.release()
    print(f"{augmentation.capitalize()} completed for {os.path.basename(input_path)}")

def augment_videos(input_folder, output_folder, crop_size, max_clips):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    augmentations = ['horizontal', 'center_crop', 'rotated', 'downsampled', 'translated', 'brightness', 'contrast']

    video_clip_names = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    no_of_clips_available = len(video_clip_names)
    
    for clip_no in range(no_of_clips_available):
        input_path = os.path.join(input_folder, video_clip_names[clip_no])
        
        with ThreadPoolExecutor() as executor:
            for i in range(max_clips):
                aug = random.choice(augmentations)
                temp = video_clip_names[clip_no].replace(" ", "").split(".")
                editted_name = temp[0] + "_" + aug + "_" + str(i) + "." + temp[1]
                output_path = os.path.join(output_folder, editted_name)
                executor.submit(augment_video, input_path, output_path, crop_size, aug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-folder-path', type=str, required=True, help='Path of folder that contains videos to be augmented')
    parser.add_argument('--output-folder-path', type=str, required=True, help='Path of folder to save augmented videos')
    parser.add_argument('--max-clips', type=int, required=True, help='Max number of clips to augment per input video')
    parser.add_argument('--crop-size', type=int, nargs=2, default=(850, 850), help='Size for center crop (height, width)')
    
    opt = parser.parse_args()
    main_folder_path = opt.main_folder_path
    output_folder_path = opt.output_folder_path
    max_clips = opt.max_clips
    crop_size = tuple(opt.crop_size)

    start_time = time.time()
    augment_videos(main_folder_path, output_folder_path, crop_size, max_clips)
    end_time = time.time()
    
    print(f"Full time by code: {end_time - start_time} seconds")
