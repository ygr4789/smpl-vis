import argparse
import os
import subprocess
from pathlib import Path

from visualize.process_pkl import process_pkl_file
from visualize.const import *

OUTPUT_DIR_PATH = Path(OUTPUT_DIR)
CACHE_DIR_PATH = Path(CACHE_DIR)
RESULT_DIR_PATH = Path(VIDEO_DIR)
RENDER_SMPL_SCRIPT = "blender/render_smpl.py"
RENDER_PRIM_SCRIPT = "blender/render_prim.py"

def render_sequence(script: str, target_flag: int, output_name: str, video_dir: str, camera_no: int, scene_no: int, soft: bool, high: bool) -> None:
    """Render a sequence using Blender."""
    cmd = [
        "blender",
        BLENDER_PATH,  # Use constant from visualize.const
        "--background",
        "--python", script,
        "--",
        "-i", str(OUTPUT_DIR_PATH / output_name),
        "-o", str(video_dir),
        "-t", str(target_flag),
        "-c", str(camera_no),
        "-sc", str(scene_no),
    ]
    
    if soft:
        cmd.append("-s")
    if high:
        cmd.append("-q")
        
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    subprocess.run(cmd, check=True, env=env)

def main() -> None:
    parser = argparse.ArgumentParser(description="Build and render SMPL meshes")
    parser.add_argument("-i", "--input", type=str, required=True, help=".pkl file or path to data directory (when using -a)")
    parser.add_argument("-a", "--ablation", action="store_true", help="Ablation dataset rendering")
    parser.add_argument("-g", "--gt", action="store_true", help="GT dataset rendering")
    
    parser.add_argument('-c', '--camera', type=int, help='Camera number, -1 for all cameras', default=-1)
    parser.add_argument('-sc', '--scene', type=int, help='Scene number, default=0 for no furnitures', default=0)
    parser.add_argument('-s', '--soft', action='store_true', help='Use soft material')
    parser.add_argument('-q', '--high', action='store_true', help='Use high quality rendering settings')
    parser.add_argument('-p', '--prim', action='store_true', help='Use primitive rendering')
    
    args = parser.parse_args()
    input_path = args.input
    ablation = args.ablation
    gt = args.gt
    
    camera_no = args.camera
    scene_no = args.scene
    soft = args.soft
    high = args.high
    prim = args.prim
    
    # Create necessary directories
    OUTPUT_DIR_PATH.mkdir(exist_ok=True)
    CACHE_DIR_PATH.mkdir(exist_ok=True)
    RESULT_DIR_PATH.mkdir(exist_ok=True)

    if not input_path:
        print("Error: Input path (-i/--input) is required")
        return

    input_path = Path(input_path)
    video_dir = os.path.join(RESULT_DIR_PATH, ('smpl_' if not prim else 'prim_') + input_path.stem)
    
    # if os.path.exists(video_dir):
    #     print(f"Video directory already exists: {video_dir}")
    #     return
    
    script = RENDER_SMPL_SCRIPT if not prim else RENDER_PRIM_SCRIPT
    
    if input_path.is_file():
        if input_path.suffix != '.pkl':
            print(f"Error: {input_path} is not a .pkl file")
            return
        if ablation:
            print(f"Error: Ablation mode is not supported for single file rendering")
            return
        
        if gt:
            process_pkl_file(str(input_path), keys_to_process_per_flag['gt'])
            render_sequence(script, TARGET_FLAG_GT, input_path.stem, video_dir, camera_no, scene_no, soft, high)
            render_sequence(script, TARGET_FLAG_INPUT, input_path.stem, video_dir, camera_no, scene_no, soft, high)
            render_sequence(script, TARGET_FLAG_REFINE, input_path.stem, video_dir, camera_no, scene_no, False, high)
        else:
            process_pkl_file(str(input_path), keys_to_process_per_flag['default'])
            render_sequence(script, TARGET_FLAG_NONE, input_path.stem, video_dir, camera_no, scene_no, soft, high)
            render_sequence(script, TARGET_FLAG_INPUT, input_path.stem, video_dir, camera_no, scene_no, soft, high)
            render_sequence(script, TARGET_FLAG_REFINE, input_path.stem, video_dir, camera_no, scene_no, False, high)
            
    elif input_path.is_dir():
        if ablation:
            pkl_suffixes = ['_all.pkl', '_wocontact.pkl', '_woprox.pkl', '_woig.pkl', '_wopose.pkl']
            pkl_files = []
            for suffix in pkl_suffixes:
                matching_files = list(input_path.glob(f'*{suffix}'))
                if not matching_files:
                    print(f"Error: No files found ending with suffix: {suffix}")
                    return
                if len(matching_files) > 1:
                    print(f"Error: Multiple files found ending with suffix: {suffix}")
                    return
                pkl_files.extend(matching_files)
                    
            file_all = pkl_files[0]
            file_wocontact = pkl_files[1]
            file_woprox = pkl_files[2]
            file_woig = pkl_files[3]
            file_wopose = pkl_files[4]
            
            process_pkl_file(str(file_all), keys_to_process_per_flag['ab_all'])
            for file_wo in [file_wocontact, file_woprox, file_woig, file_wopose]:
                process_pkl_file(str(file_wo), keys_to_process_per_flag['ab_wo'])
            
            render_sequence(script, TARGET_FLAG_GT, file_all.stem, video_dir, camera_no, scene_no, soft, high)
            render_sequence(script, TARGET_FLAG_PSEUDO_GT, file_all.stem, video_dir, camera_no, scene_no, soft, high)
            render_sequence(script, TARGET_FLAG_REFINE_PSEUDO_GT, file_all.stem, video_dir, camera_no, scene_no, False, high)
            render_sequence(script, TARGET_FLAG_WOCONTACT, file_wocontact.stem, video_dir, camera_no, scene_no, False, high)
            render_sequence(script, TARGET_FLAG_WOPROX, file_woprox.stem, video_dir, camera_no, scene_no, False, high)
            render_sequence(script, TARGET_FLAG_WOIG, file_woig.stem, video_dir, camera_no, scene_no, False, high)
            render_sequence(script, TARGET_FLAG_WOPOSE, file_wopose.stem, video_dir, camera_no, scene_no, False, high)
            
        else:
            print("Error: Directory input is only available with ablation mode (-a/--ablation)")
            return
        
    else:
        print(f"Error: {input_path} does not exist")
        return

if __name__ == "__main__":
    main() 