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
        # 일반적으로 Linux에서는 DejaVuSans-Bold.ttf가 존재합니다.
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    return font

def make_grid_with_labels(images, labels, grid_shape=(2, 4)):
    """각 타일의 왼쪽 상단에 큰 글씨로 모드 이름을 표시하고 grid (rows, cols)로 이미지를 배열합니다."""
    font = get_font(40)  # 타일 내 모드 이름 폰트 크기
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
                # 남은 칸은 검정색 이미지로 채움
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

    modes = ["all", "gt", "gt_input", "wocontact", "woig", "wopose", "woprox"]
    cam_indices = list(range(6))

    for cam in cam_indices:
        key_prefixes = [f"{mode}_cam{cam:02d}" for mode in modes]
        available_keys = [k for k in key_prefixes if (input_dir / f"{k}.mp4").exists()]
        if not available_keys:
            continue

        # 각 모드에 대해 프레임 추출
        video_frames_cam = {k: extract_frames(input_dir / f"{k}.mp4") for k in available_keys}
        # 가장 짧은 비디오의 프레임 수만 사용
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
            # 그리드 구성: 2행 4열 (모드 개수 7개이므로 8번째는 빈 칸(검정)으로 처리)
            concat_image = make_grid_with_labels(images, labels, grid_shape=(2, 4))
            concat_pil = Image.fromarray(concat_image)

            # 전체 프레임 번호를 큰 글씨로 표시 (왼쪽 상단)
            draw = ImageDraw.Draw(concat_pil)
            frame_font = get_font(50)
            draw.text((10, 10), f"Frame {i:03d}", fill="white", font=frame_font)

            concat_frames.append(np.array(concat_pil))

        # 출력 비디오 작성
        height, width, _ = concat_frames[0].shape
        output_path = output_dir / f"concat_cam{cam:02d}.mp4"
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for frame in concat_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    return output_dir

# 실행 예제:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and render SMPL meshes")
    parser.add_argument("--folder", type=str)
    args = parser.parse_args()
    # process_video_folder_with_labels("ablation/prim_2_2")
    process_video_folder_with_labels(args.folder)
