import os
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse

def extract_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

def get_font(size=40):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()

def process_video_folder_with_labels(subdir, fps=30):
    root_dir = Path("./video") /  subdir
    output_dir = Path("./video") / "concat" / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    keyword_folders = sorted([
        f for f in root_dir.iterdir()
        if f.is_dir() and f.name.startswith("prim_pseudo_gt_")
    ])

    cam_indices = list(range(6))

    for cam in cam_indices:
        images_per_frame = []
        font = get_font(40)

        for keyword_folder in keyword_folders:
            folder_name = keyword_folder.name
            keyword = folder_name.split("_")[-2] if folder_name.endswith("_vis") else folder_name

            # if "orig" in folder_name:
            #     video_path = keyword_folder / f"input_cam{cam:02d}.mp4"
            #     label_prefix = f"INPUT-{keyword}"
            # else:
            video_path = keyword_folder / f"refine_cam{cam:02d}.mp4"
            label_prefix = f"REFINE-{keyword}"

            if not video_path.exists():
                continue

            frames = extract_frames(video_path)
            images_per_frame.append((label_prefix, frames))

        if not images_per_frame:
            continue

        # Align all to the minimum number of frames
        min_len = min(len(frames) for _, frames in images_per_frame)
        concat_frames = []

        for i in range(min_len):
            stacked_imgs = []
            for label, frames in images_per_frame:
                img = frames[i].copy()
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), label, fill="white", font=font)
                stacked_imgs.append(np.asarray(img))
            frame_grid = np.vstack(stacked_imgs)
            concat_frames.append(frame_grid)

        height, width, _ = concat_frames[0].shape
        output_path = output_dir / f"cam{cam:02d}.mp4"
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for frame in concat_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    return root_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate videos for each camera")
    parser.add_argument("--folder", type=str, required=True, help="e.g., obj_var_48_6")
    args = parser.parse_args()
    process_video_folder_with_labels(args.folder)
