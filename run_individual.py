import os
import subprocess
import glob
import shlex
import sys

# Configuration
TEST_DIR = "/workspace/test"
OUTPUT_DIR = "/workspace/results_test_check"
BASE_GEN = "/workspace/IMTalker/checkpoints/generator.ckpt"
BASE_REND = "/workspace/IMTalker/checkpoints/renderer.ckpt"
FT_GEN = "/workspace/working_code/IMTalker/imtalker_checkpoints_finetunned/generator.ckpt"
FT_REND = "/workspace/working_code/IMTalker/imtalker_checkpoints_finetunned/renderer.ckpt"
PYTHON_BIN = "/opt/conda/envs/IMTalker/bin/python"
GEN_SCRIPT = "/workspace/IMTalker/generator/generate.py"
WAV2VEC_PATH = "/workspace/IMTalker/checkpoints/wav2vec2-base-960h"
FFMPEG_BIN = "/opt/conda/envs/IMTalker/bin/ffmpeg"

# Ensure environment is shared with all children
os.environ["PYTHONPATH"] = f"/workspace/IMTalker:{os.environ.get('PYTHONPATH', '')}"
os.environ["PATH"] = f"/opt/conda/envs/IMTalker/bin:{os.environ.get('PATH', '')}"

def run_cmd(cmd):
    print(f"\n[EXEC] {cmd}")
    # Force flushing of stdout
    sys.stdout.flush()
    result = subprocess.run(cmd, shell=True, capture_output=False, cwd="/workspace/IMTalker")
    return result.returncode

def main():
    images = sorted([f for f in glob.glob(os.path.join(TEST_DIR, "*")) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
    audios = sorted([f for f in glob.glob(os.path.join(TEST_DIR, "*.wav"))])

    print(f"Total Images: {len(images)}")
    print(f"Total Audios: {len(audios)}")
    print(f"Expected Pairs: {len(images) * len(audios)}")

    os.makedirs(os.path.join(OUTPUT_DIR, "base"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "finetuned"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "merged"), exist_ok=True)

    count = 0
    success_count = 0

    for img in images:
        img_basename = os.path.basename(img)
        img_name = os.path.splitext(img_basename)[0]
        
        for aud in audios:
            aud_basename = os.path.basename(aud)
            aud_name = os.path.splitext(aud_basename)[0]
            count += 1
            
            comb_name = f"{img_name}_{aud_name}"
            print(f"\n=== PAIR {count}/{len(images)*len(audios)}: {img_basename} + {aud_basename} ===")
            
            base_vid = os.path.join(OUTPUT_DIR, "base", f"{comb_name}.mp4")
            ft_vid = os.path.join(OUTPUT_DIR, "finetuned", f"{comb_name}.mp4")
            merged_vid = os.path.join(OUTPUT_DIR, "merged", f"{comb_name}_merged.mp4")

            # Skip if already merged
            if os.path.exists(merged_vid):
                print(f"Already exists: {merged_vid}")
                success_count += 1
                continue

            # 1. Base Model Inference
            if not os.path.exists(base_vid):
                # Script saves as [img_name].mp4 in res_dir
                cmd_base = (f"{PYTHON_BIN} {shlex.quote(GEN_SCRIPT)} "
                            f"--ref_path {shlex.quote(img)} --aud_path {shlex.quote(aud)} "
                            f"--res_dir {shlex.quote(os.path.join(OUTPUT_DIR, 'base'))} "
                            f"--generator_path {shlex.quote(BASE_GEN)} --renderer_path {shlex.quote(BASE_REND)} "
                            f"--wav2vec_model_path {shlex.quote(WAV2VEC_PATH)} "
                            f"--a_cfg_scale 2 --crop")
                ret = run_cmd(cmd_base)
                
                # Check if file was created at expected default name
                default_name = os.path.join(OUTPUT_DIR, "base", f"{img_name}.mp4")
                if os.path.exists(default_name):
                    os.rename(default_name, base_vid)
                else:
                    print(f"FAILED to generate BASE video for {comb_name}")
                    continue

            # 2. Fine-tuned Model Inference
            if not os.path.exists(ft_vid):
                cmd_ft = (f"{PYTHON_BIN} {shlex.quote(GEN_SCRIPT)} "
                           f"--ref_path {shlex.quote(img)} --aud_path {shlex.quote(aud)} "
                           f"--res_dir {shlex.quote(os.path.join(OUTPUT_DIR, 'finetuned'))} "
                           f"--generator_path {shlex.quote(FT_GEN)} --renderer_path {shlex.quote(FT_REND)} "
                           f"--wav2vec_model_path {shlex.quote(WAV2VEC_PATH)} "
                           f"--a_cfg_scale 2 --crop")
                ret = run_cmd(cmd_ft)
                
                default_name = os.path.join(OUTPUT_DIR, "finetuned", f"{img_name}.mp4")
                if os.path.exists(default_name):
                    os.rename(default_name, ft_vid)
                else:
                    print(f"FAILED to generate FINETUNED video for {comb_name}")
                    continue

            # 3. Merge Side-by-Side
            if os.path.exists(base_vid) and os.path.exists(ft_vid):
                print(f"Merging into: {merged_vid}")
                cmd_merge = (f"{FFMPEG_BIN} -i {shlex.quote(base_vid)} -i {shlex.quote(ft_vid)} "
                             f"-filter_complex \"[0:v]drawtext=text='Base':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[v0]; "
                             f"[1:v]drawtext=text='Fine-tuned':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[v1]; "
                             f"[v0][v1]hstack=inputs=2[v]\" "
                             f"-map \"[v]\" -map 1:a -c:v libx264 -crf 18 -c:a copy {shlex.quote(merged_vid)} -y")
                ret = run_cmd(cmd_merge)
                if ret == 0:
                    success_count += 1
                else:
                    print(f"FAILED to merge {comb_name}")

    print(f"\nSequential Processing Finished!")
    print(f"Successful Merges: {success_count}/{len(images)*len(audios)}")

if __name__ == "__main__":
    main()
