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
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    return font

def make_grid_with_labels(images, labels, grid_shape=(1, 3)):
    """각 타일의 왼쪽 상단에 큰 글씨로 모드 이름을 표시하고 grid (rows, cols)로 이미지를 배열합니다."""
    font = get_font(40)
    rows = []
    for i in range(grid_shape[0]):
        row_imgs = []
        for j in range(grid_shape[1]):
            idx = i * grid_shape[1] + j
            if idx < len(images):
                img = images[idx].copy()
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), labels[idx], fill="white", font=font)
                row_imgs.append(np.asarray(img))
            else:
                w, h = images[0].size
                black = Image.new("RGB", (w, h), (0, 0, 0))
                row_imgs.append(np.asarray(black))
        row = np.hstack(row_imgs)
        rows.append(row)
    return np.vstack(rows)

def process_video_folder_with_labels(subdir, fps=30):
    base_dir = Path("./video")
    input_dir = base_dir / subdir
    output_dir = base_dir / "concat" / subdir.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = ["gt", "input", "refine"]
    cam_indices = list(range(6))

    for cam in cam_indices:
        key_prefixes = [f"{mode}_cam{cam:02d}" for mode in modes]
        available_keys = [k for k in key_prefixes if (input_dir / f"{k}.mp4").exists()]
        if not available_keys:
            continue

        video_frames_cam = {k: extract_frames(input_dir / f"{k}.mp4") for k in available_keys}
        num_frames = min(len(frames) for frames in video_frames_cam.values())
        concat_frames = []

        for i in range(num_frames):
            images = []
            labels = []
            for mode in modes:
                key = f"{mode}_cam{cam:02d}"
                if key in video_frames_cam:
                    images.append(video_frames_cam[key][i])
                    labels.append(mode)
            concat_image = make_grid_with_labels(images, labels, grid_shape=(1, 3))
            concat_pil = Image.fromarray(concat_image)

            draw = ImageDraw.Draw(concat_pil)
            frame_font = get_font(50)
            draw.text((10, 10), f"Frame {i:03d}", fill="white", font=frame_font)

            concat_frames.append(np.array(concat_pil))

        height, width, _ = concat_frames[0].shape
        output_path = output_dir / f"concat_cam{cam:02d}.mp4"
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for frame in concat_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and render SMPL meshes")
    parser.add_argument("--folder", type=str)
    args = parser.parse_args()
    process_video_folder_with_labels(args.folder)
