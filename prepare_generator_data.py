"""
prepare_generator_data.py — Multi-GPU parallel version

Extracts features needed for IMTalker's generator training:
  - motion/{video_name}.pt   -> Motion latents (N, 32)
  - audio/{video_name}.npy   -> Wav2Vec2 features (N, 768)
  - smirk/{video_name}.pt    -> Pose dict: pose_params (N,3), cam (N,3)
  - gaze/{video_name}.npy    -> Gaze direction (N, 2)

Usage:
    python prepare_generator_data.py \
        --renderer_data_dir /workspace/Final_Data/renderer_dataset \
        --mp4_dir /workspace/Final_Data/mp4 \
        --output_dir /workspace/Final_Data/generator_dataset \
        --renderer_ckpt ./IMTalker/checkpoints/renderer.ckpt \
        --wav2vec_path ./IMTalker/checkpoints/wav2vec2-base-960h
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from glob import glob
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import torchvision.transforms as transforms

IMTALKER_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IMTalker")
sys.path.insert(0, IMTALKER_ROOT)


# ── Pose / Gaze estimation (CPU-only, no imports needed) ──────────────

def estimate_pose_from_landmarks(lmd_path, num_frames):
    if lmd_path and os.path.exists(lmd_path):
        with open(lmd_path, 'r') as f:
            lines = f.readlines()
        lines.sort()
        poses, cams = [], []
        for line in lines:
            parts = [c for c in line.strip().split(' ') if c]
            if len(parts) < 69:
                poses.append([0., 0., 0.]); cams.append([0., 0., 0.]); continue
            coords = parts[1:]
            lm = []
            for coord in coords:
                x, y = coord.split('_')
                lm.append([float(x), float(y)])
            lm = np.array(lm)
            if len(lm) < 68:
                poses.append([0., 0., 0.]); cams.append([0., 0., 0.]); continue
            nose = lm[30]; left_face = lm[0]; right_face = lm[16]
            chin = lm[8]; forehead = (lm[19] + lm[24]) / 2
            fw = np.linalg.norm(right_face - left_face) + 1e-6
            fh = np.linalg.norm(chin - forehead) + 1e-6
            fcx = (left_face[0] + right_face[0]) / 2
            fcy = (forehead[1] + chin[1]) / 2
            yaw = (nose[0] - fcx) / (fw / 2)
            pitch = (nose[1] - fcy) / (fh / 2)
            le = lm[36:42].mean(0); re = lm[42:48].mean(0)
            roll = float(np.arctan2(re[1] - le[1], re[0] - le[0]))
            poses.append([float(pitch), float(yaw), roll])
            cams.append([float(fw / 512.), float(fcx / 512. - .5), float(fcy / 512. - .5)])
        poses = poses[:num_frames]; cams = cams[:num_frames]
        while len(poses) < num_frames:
            poses.append(poses[-1] if poses else [0, 0, 0])
            cams.append(cams[-1] if cams else [0, 0, 0])
        return torch.tensor(poses, dtype=torch.float32), torch.tensor(cams, dtype=torch.float32)
    return torch.zeros(num_frames, 3), torch.zeros(num_frames, 3)


def estimate_gaze_from_landmarks(lmd_path, num_frames):
    if lmd_path and os.path.exists(lmd_path):
        with open(lmd_path, 'r') as f:
            lines = f.readlines()
        lines.sort()
        gazes = []
        for line in lines:
            parts = [c for c in line.strip().split(' ') if c]
            if len(parts) < 69:
                gazes.append([0., 0.]); continue
            coords = parts[1:]
            lm = []
            for coord in coords:
                x, y = coord.split('_')
                lm.append([float(x), float(y)])
            lm = np.array(lm)
            if len(lm) < 68:
                gazes.append([0., 0.]); continue
            ec = ((lm[36:42].mean(0) + lm[42:48].mean(0)) / 2)
            nose = lm[30]
            gazes.append([float((ec[1] - nose[1]) / 50.), float((ec[0] - nose[0]) / 50.)])
        gazes = gazes[:num_frames]
        while len(gazes) < num_frames:
            gazes.append(gazes[-1] if gazes else [0, 0])
        return np.array(gazes, dtype=np.float32)
    return np.zeros((num_frames, 2), dtype=np.float32)


def extract_audio_from_mp4(mp4_path, output_wav_path):
    cmd = ["ffmpeg", "-i", str(mp4_path), "-vn", "-acodec", "pcm_s16le",
           "-ar", "16000", "-ac", "1", str(output_wav_path), "-y", "-loglevel", "error"]
    return subprocess.run(cmd, capture_output=True).returncode == 0


# ── GPU Worker ────────────────────────────────────────────────────────

def process_video(video_name, frame_dir, mp4_dir, lmd_root, output_dir,
                  renderer, wav2vec, preprocessor, transform,
                  device, batch_size, gpu_lock, smirk_dir=None, l2cs_dir=None):
    """Process a single video: extract motion, audio, pose, gaze."""
    motion_path = Path(output_dir) / "motion" / f"{video_name}.pt"
    audio_path = Path(output_dir) / "audio" / f"{video_name}.npy"
    smirk_path = Path(output_dir) / "smirk" / f"{video_name}.pt"
    gaze_path = Path(output_dir) / "gaze" / f"{video_name}.npy"

    if motion_path.exists() and audio_path.exists() and smirk_path.exists() and gaze_path.exists():
        return "skipped"

    # 1. Motion latents (GPU)
    frame_files = sorted(
        list(Path(frame_dir).glob("*.jpg")) + list(Path(frame_dir).glob("*.png")),
        key=lambda x: x.name
    )
    if len(frame_files) < 10:
        return "too_few_frames"

    # Pre-load frames on CPU
    batch_tensors = []
    for fp in frame_files:
        img = Image.open(fp).convert("RGB")
        batch_tensors.append(transform(img))

    # GPU: extract motion in batches
    all_latents = []
    with gpu_lock:
        for i in range(0, len(batch_tensors), batch_size):
            batch = torch.stack(batch_tensors[i:i + batch_size]).to(device)
            with torch.no_grad():
                latents = renderer.latent_token_encoder(batch)
            all_latents.append(latents.cpu())

    motion_latents = torch.cat(all_latents, dim=0)
    num_frames = len(motion_latents)

    # 2. Audio features (GPU)
    mp4_path = Path(mp4_dir) / f"{video_name}.mp4"
    if not mp4_path.exists():
        return "no_mp4"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
        if not extract_audio_from_mp4(mp4_path, tmp_wav.name):
            return "audio_fail"
        import librosa
        speech_array, sr = librosa.load(tmp_wav.name, sr=16000)

    input_values = preprocessor(
        speech_array, sampling_rate=16000, return_tensors='pt'
    ).input_values[0].unsqueeze(0)

    with gpu_lock:
        with torch.no_grad():
            output = wav2vec(input_values.to(device), seq_len=num_frames)
        audio_features = output.last_hidden_state.squeeze(0).cpu().numpy()

    # 3. Pose (CPU)
    if smirk_dir and os.path.exists(os.path.join(smirk_dir, f"{video_name}.pt")):
        sd = torch.load(os.path.join(smirk_dir, f"{video_name}.pt"))
        pose_params, cam = sd["pose_params"][:num_frames], sd["cam"][:num_frames]
    else:
        lmd_path = str(Path(lmd_root) / f"{video_name}.txt")
        pose_params, cam = estimate_pose_from_landmarks(lmd_path, num_frames)

    # 4. Gaze (CPU)
    if l2cs_dir and os.path.exists(os.path.join(l2cs_dir, f"{video_name}.npy")):
        gaze = np.load(os.path.join(l2cs_dir, f"{video_name}.npy"))[:num_frames]
    else:
        lmd_path = str(Path(lmd_root) / f"{video_name}.txt")
        gaze = estimate_gaze_from_landmarks(lmd_path, num_frames)

    # 5. Save
    torch.save(motion_latents, motion_path)
    np.save(audio_path, audio_features)
    torch.save({"pose_params": pose_params, "cam": cam}, smirk_path)
    np.save(gaze_path, gaze)

    return "ok"


def gpu_worker(gpu_id, video_shard, args_dict, progress_dict):
    """One process per GPU with threaded CPU/IO overlap."""
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    try:
        from renderer.models import IMTRenderer
        from generator.wav2vec2 import Wav2VecModel
        from transformers import Wav2Vec2FeatureExtractor
        import librosa
    except Exception as e:
        print(f"[GPU {gpu_id}] Import Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load renderer
    class RendererArgs:
        depth = 2; latent_dim = 32; swin_res_threshold = 128
        num_heads = 8; window_size = 8; drop_path = 0.1; low_res_depth = 2

    renderer = IMTRenderer(RendererArgs()).to(device)
    ckpt = torch.load(args_dict["renderer_ckpt"], map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    clean = {k.replace("gen.", ""): v for k, v in sd.items() if k.startswith("gen.")}
    renderer.load_state_dict(clean or sd, strict=False)
    renderer.eval()

    # Load wav2vec
    wav2vec = Wav2VecModel.from_pretrained(args_dict["wav2vec_path"], local_files_only=True).to(device)
    wav2vec.eval()
    wav2vec.feature_extractor._freeze_parameters()
    for p in wav2vec.parameters():
        p.requires_grad = False
    preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(args_dict["wav2vec_path"], local_files_only=True)

    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

    gpu_lock = threading.Lock()
    success, skipped, failed = 0, 0, 0
    done_count = 0

    def _process(item):
        nonlocal success, skipped, failed, done_count
        video_name, frame_dir = item
        try:
            status = process_video(
                video_name, frame_dir,
                args_dict["mp4_dir"], args_dict["lmd_root"], args_dict["output_dir"],
                renderer, wav2vec, preprocessor, transform,
                device, args_dict["batch_size"], gpu_lock,
                args_dict.get("smirk_dir"), args_dict.get("l2cs_dir")
            )
            if status == "skipped":
                skipped += 1
            elif status == "ok":
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        done_count += 1
        progress_dict[gpu_id] = done_count

    threads = args_dict.get("threads_per_gpu", 4)
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(_process, item) for item in video_shard]
        for f in as_completed(futures):
            pass

    print(f"[GPU {gpu_id}] Done. Success={success} Skipped={skipped} Failed={failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer_data_dir", type=str, required=True)
    parser.add_argument("--mp4_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--renderer_ckpt", type=str, default="./IMTalker/checkpoints/renderer.ckpt")
    parser.add_argument("--wav2vec_path", type=str, default="./IMTalker/checkpoints/wav2vec2-base-960h")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--threads_per_gpu", type=int, default=4)
    parser.add_argument("--smirk_dir", type=str, default=None)
    parser.add_argument("--l2cs_dir", type=str, default=None)
    parser.add_argument("--max_videos", type=int, default=None)
    args = parser.parse_args()

    num_gpus = args.num_gpus or torch.cuda.device_count()

    for sub in ["motion", "audio", "smirk", "gaze"]:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    video_frame_root = Path(args.renderer_data_dir) / "video_frame"
    lmd_root = str(Path(args.renderer_data_dir) / "lmd")
    video_dirs = sorted([d for d in video_frame_root.iterdir() if d.is_dir()])
    if args.max_videos:
        video_dirs = video_dirs[:args.max_videos]

    items = [(d.name, str(d)) for d in video_dirs]
    total = len(items)

    print(f"Videos: {total}")
    print(f"GPUs: {num_gpus} x {args.threads_per_gpu} threads = {num_gpus * args.threads_per_gpu} workers")
    print(f"Output: {args.output_dir}")
    print()

    # Split round-robin
    shards = [[] for _ in range(num_gpus)]
    for i, item in enumerate(items):
        shards[i % num_gpus].append(item)

    args_dict = {
        "mp4_dir": args.mp4_dir,
        "output_dir": args.output_dir,
        "renderer_ckpt": args.renderer_ckpt,
        "wav2vec_path": args.wav2vec_path,
        "batch_size": args.batch_size,
        "lmd_root": lmd_root,
        "threads_per_gpu": args.threads_per_gpu,
        "smirk_dir": args.smirk_dir,
        "l2cs_dir": args.l2cs_dir,
    }

    manager = mp.Manager()
    progress_dict = manager.dict({i: 0 for i in range(num_gpus)})

    mp.set_start_method("spawn", force=True)
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=gpu_worker, args=(gpu_id, shards[gpu_id], args_dict, progress_dict))
        p.start()
        processes.append(p)

    pbar = tqdm(total=total, desc=f"All GPUs ({num_gpus * args.threads_per_gpu} workers)")
    while any(p.is_alive() for p in processes):
        done = sum(progress_dict.values())
        pbar.n = min(done, total)
        pbar.refresh()
        time.sleep(2)

    pbar.n = total
    pbar.refresh()
    pbar.close()

    for p in processes:
        p.join()

    print(f"\nDone! Dataset at: {args.output_dir}")


if __name__ == "__main__":
    main()
