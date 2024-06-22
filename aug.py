import os
import cv2
import random
import numpy as np
import argparse

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

def apply_augmentations(frames, aug, width, height):
    if aug == 'rotated':
        angle = random.randint(-10, 10)
        M = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
        frames = [cv2.warpAffine(frame, M, (width, height)) for frame in frames]
    elif aug == 'brightness':
        factor = random.uniform(0.5, 1.5)
        frames = [cv2.convertScaleAbs(frame, alpha=factor, beta=0) for frame in frames]
    elif aug == 'contrast':
        factor = random.uniform(0.5, 1.5)
        frames = [cv2.convertScaleAbs(frame, alpha=factor, beta=0) for frame in frames]
    elif aug == 'translation':
        max_trans = 10  # Max pixels to translate
        tx = random.randint(-max_trans, max_trans)
        ty = random.randint(-max_trans, max_trans)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        frames = [cv2.warpAffine(frame, M, (width, height)) for frame in frames]
    return frames

def augment_videos(main_folder_path, output_folder_path, crop_size):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    augmentations = ['rotated', 'brightness', 'contrast', 'translation']
    center_crop = CenterCrop(crop_size)

    for filename in os.listdir(main_folder_path):
        if filename.endswith(".MOV") or filename.endswith(".MP4"):
            input_path = os.path.join(main_folder_path, filename)
            
            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            # Apply center crop to the video
            frames = center_crop(frames)
            crop_h, crop_w = crop_size
            cropped_output_path = os.path.join(output_folder_path, f"center_crop_{filename}")
            out = cv2.VideoWriter(cropped_output_path, fourcc, fps, (crop_w, crop_h))
            for frame in frames:
                out.write(frame)
            out.release()
            print(f"Center crop completed for {filename}")

            # Apply other augmentations on the cropped video
            for aug in augmentations:
                output_filename = f"{aug}_{filename}"
                output_path = os.path.join(output_folder_path, output_filename)
                
                out = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))
                augmented_frames = apply_augmentations(frames, aug, crop_w, crop_h)
                for frame in augmented_frames:
                    out.write(frame)
                out.release()
                print(f"{aug.capitalize()} completed for {filename}")

            cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-folder-path', type=str, required=True, help='Path of folder that contains video clips to be augmented')
    parser.add_argument('--output-folder-path', type=str, required=True, help='Path of folder that will contain augmented video clips')
    parser.add_argument('--crop-size', type=int, nargs=2, default=[1000, 1000], help='Crop size for center cropping in the format (height, width)')

    args = parser.parse_args()

    main_folder_path = args.main_folder_path
    output_folder_path = args.output_folder_path
    crop_size = tuple(args.crop_size)

    augment_videos(main_folder_path, output_folder_path, crop_size)

if __name__ == '__main__':
    main()
