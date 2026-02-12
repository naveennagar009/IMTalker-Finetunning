import os
import subprocess
import glob
import shlex
import sys

# Configuration
TEST_DIR = "/workspace/test2"
OUTPUT_DIR = "/workspace/results_test2"
BASE_GEN = "/workspace/IMTalker/checkpoints/generator.ckpt"
BASE_REND = "/workspace/IMTalker/checkpoints/renderer.ckpt"
FT_GEN = "/workspace/exps/custom_generator_v1/checkpoints/step=026000.ckpt"
FT_REND = "/workspace/IMTalker/exps/custom_renderer_v2/checkpoints/batch=step=030000.ckpt"
PYTHON_BIN = "/opt/conda/envs/IMTalker/bin/python"
GEN_SCRIPT = "/workspace/IMTalker/generator/generate.py"
WAV2VEC_PATH = "/workspace/IMTalker/checkpoints/wav2vec2-base-960h"
FFMPEG_BIN = "/opt/conda/envs/IMTalker/bin/ffmpeg"

# Ensure environment is shared with all children
os.environ["PYTHONPATH"] = f"/workspace/IMTalker:{os.environ.get('PYTHONPATH', '')}"
os.environ["PATH"] = f"/opt/conda/envs/IMTalker/bin:{os.environ.get('PATH', '')}"

images = [f for f in glob.glob(os.path.join(TEST_DIR, "*")) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
audios = [f for f in glob.glob(os.path.join(TEST_DIR, "*.wav"))]

os.makedirs(os.path.join(OUTPUT_DIR, "base"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "finetuned"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "merged"), exist_ok=True)

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    # Capture output to diagnose failures
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/workspace/IMTalker")
    if result.returncode != 0:
        print(f"COMMAND FAILED (Code {result.returncode}):")
        print(f"STDOUT: {result.stdout[-500:]}")
        print(f"STDERR: {result.stderr[-500:]}")
    return result.returncode

failed_images = set()

print(f"Found {len(images)} images and {len(audios)} audios.")

for i, img in enumerate(sorted(images)):
    img_name = os.path.splitext(os.path.basename(img))[0]
    if img_name in failed_images:
        continue
        
    print(f"\n--- Image {i+1}/{len(images)}: {img_name} ---")
    
    for j, aud in enumerate(sorted(audios)):
        aud_name = os.path.splitext(os.path.basename(aud))[0]
        comb_name = f"{img_name}_{aud_name}"
        
        base_vid = os.path.join(OUTPUT_DIR, "base", f"{comb_name}.mp4")
        ft_vid = os.path.join(OUTPUT_DIR, "finetuned", f"{comb_name}.mp4")
        merged_vid = os.path.join(OUTPUT_DIR, "merged", f"{comb_name}_merged.mp4")
        
        # 1. Base Inference
        if not os.path.exists(base_vid):
            cmd_base = (f"{PYTHON_BIN} {shlex.quote(GEN_SCRIPT)} "
                        f"--ref_path {shlex.quote(img)} --aud_path {shlex.quote(aud)} "
                        f"--res_dir {shlex.quote(os.path.join(OUTPUT_DIR, 'base'))} "
                        f"--generator_path {shlex.quote(BASE_GEN)} --renderer_path {shlex.quote(BASE_REND)} "
                        f"--wav2vec_model_path {shlex.quote(WAV2VEC_PATH)} "
                        f"--a_cfg_scale 2 --crop")
            run_cmd(cmd_base)
            orig_saved = os.path.join(OUTPUT_DIR, "base", f"{img_name}.mp4")
            if os.path.exists(orig_saved):
                os.rename(orig_saved, base_vid)
            else:
                print(f"FAILED Base inference for {img_name}. Skipping image.")
                failed_images.add(img_name)
                break 

        # 2. Fine-tuned Inference
        if not os.path.exists(ft_vid):
            cmd_ft = (f"{PYTHON_BIN} {shlex.quote(GEN_SCRIPT)} "
                       f"--ref_path {shlex.quote(img)} --aud_path {shlex.quote(aud)} "
                       f"--res_dir {shlex.quote(os.path.join(OUTPUT_DIR, 'finetuned'))} "
                       f"--generator_path {shlex.quote(FT_GEN)} --renderer_path {shlex.quote(FT_REND)} "
                       f"--wav2vec_model_path {shlex.quote(WAV2VEC_PATH)} "
                       f"--a_cfg_scale 2 --crop")
            run_cmd(cmd_ft)
            orig_saved = os.path.join(OUTPUT_DIR, "finetuned", f"{img_name}.mp4")
            if os.path.exists(orig_saved):
                os.rename(orig_saved, ft_vid)
            else:
                print(f"FAILED Fine-tuned inference for {img_name}. Skipping image.")
                failed_images.add(img_name)
                break 

        # 3. Merge
        if os.path.exists(base_vid) and os.path.exists(ft_vid) and not os.path.exists(merged_vid):
            print(f"  Merging: {comb_name}")
            cmd_merge = (f"{FFMPEG_BIN} -i {shlex.quote(base_vid)} -i {shlex.quote(ft_vid)} "
                         f"-filter_complex \"[0:v]drawtext=text='Base':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[v0]; "
                         f"[1:v]drawtext=text='Fine-tuned':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[v1]; "
                         f"[v0][v1]hstack=inputs=2[v]\" "
                         f"-map \"[v]\" -map 1:a -c:v libx264 -crf 18 -c:a copy {shlex.quote(merged_vid)} -y")
            run_cmd(cmd_merge)

print("\nBatch processing complete!")
