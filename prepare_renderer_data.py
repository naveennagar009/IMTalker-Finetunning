"""
prepare_renderer_data.py — Multi-GPU + Multi-Thread parallel version (optimized)

Key optimizations over v1:
  - Batched landmark extraction (16 frames per GPU lock acquisition)
  - Minimal lock scope — CPU/IO work never holds the GPU lock
  - Direct ffmpeg output (no temp dir copy)
  - Multi-threaded ffmpeg decoding

Usage:
    conda activate IMTalker
    python prepare_renderer_data.py \
        --input_dir /workspace/liveavatar_data/mp4 \
        --output_dir /workspace/liveavatar_data/renderer_dataset
"""

import os
import sys
import cv2
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
from pathlib import Path
from glob import glob
from tqdm import tqdm
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_stable_face_crop_fast(video_path, fa, gpu_lock, max_samples=3):
    """Fast face crop detection — sample only a few frames.
    GPU lock is only held during the actual detection call."""
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        return None

    sample_indices = set()
    for i in range(max_samples):
        sample_indices.add(int(i * total_frames / max_samples))

    # Read sampled frames (CPU — no lock)
    sampled_frames = []
    idx = 0
    while cap.isOpened():
        ret = cap.grab()
        if not ret:
            break
        if idx in sample_indices:
            ret, frame = cap.retrieve()
            if ret:
                sampled_frames.append(frame)
        idx += 1
    cap.release()

    if not sampled_frames:
        return None

    # Detect faces (GPU — lock only here)
    bboxes_list = []
    with gpu_lock:
        for frame in sampled_frames:
            try:
                detected = fa.face_detector.detect_from_image(frame)
                if detected is not None:
                    for d in detected:
                        x1, y1, x2, y2, score = d
                        if score > 0.5:
                            bboxes_list.append([int(x1), int(y1), int(x2), int(y2)])
            except Exception:
                pass

    if not bboxes_list:
        return None

    cx = np.median([(b[0] + b[2]) / 2 for b in bboxes_list])
    cy = np.median([(b[1] + b[3]) / 2 for b in bboxes_list])
    med_w = np.median([b[2] - b[0] for b in bboxes_list])
    med_h = np.median([b[3] - b[1] for b in bboxes_list])

    crop_size = min(int(max(med_w, med_h) * 1.6), width, height)

    x1 = max(0, int(cx - crop_size / 2))
    y1 = max(0, int(cy - crop_size / 2))
    if x1 + crop_size > width:
        x1 = width - crop_size
    if y1 + crop_size > height:
        y1 = height - crop_size

    return x1, y1, crop_size, crop_size


def extract_frames_ffmpeg(video_path, frame_dir, target_fps, target_size, x1, y1, cw, ch):
    """CPU-only: extract cropped frames via ffmpeg — writes directly to output dir."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-threads", "1", "-i", str(video_path),
        "-vf", f"fps={target_fps},crop={cw}:{ch}:{x1}:{y1},scale={target_size}:{target_size}:flags=fast_bilinear",
        "-q:v", "3",
        str(frame_dir / "image_%04d.jpg"),
        "-y", "-loglevel", "error"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return sorted(frame_dir.glob("image_*.jpg"))


BATCH_LMD = 16  # frames per GPU lock acquisition


def extract_landmarks_batch(frame_paths, fa, target_size, gpu_lock):
    """Extract landmarks with batched GPU lock acquisition."""
    known_bbox = [(30, 30, target_size - 30, target_size - 30)]
    lmd_lines = []

    # Read all frames (CPU — no lock)
    frames = []
    valid_paths = []
    for fp in frame_paths:
        frame = cv2.imread(str(fp))
        if frame is not None:
            frames.append(frame)
            valid_paths.append(fp)

    # Process in batches under the lock
    for batch_start in range(0, len(frames), BATCH_LMD):
        batch_frames = frames[batch_start:batch_start + BATCH_LMD]
        batch_paths = valid_paths[batch_start:batch_start + BATCH_LMD]

        batch_results = []
        with gpu_lock:
            for frame in batch_frames:
                try:
                    lm = fa.get_landmarks_from_image(frame, detected_faces=known_bbox)
                except Exception:
                    lm = None
                batch_results.append(lm)

        for lm_result, fp in zip(batch_results, batch_paths):
            if lm_result is None or len(lm_result) == 0:
                lmd_str = " ".join(["0_0"] * 68)
            else:
                lm = lm_result[0]
                coords = [f"{lm[j, 0]:.1f}_{lm[j, 1]:.1f}" for j in range(68)]
                lmd_str = " ".join(coords)
            lmd_lines.append(f"{fp.name} {lmd_str}")

    return lmd_lines


def process_single_video(video_path, output_dir, target_fps, target_size, fa, gpu_lock):
    """Full pipeline for one video."""
    video_name = Path(video_path).stem
    frame_dir = Path(output_dir) / "video_frame" / video_name
    lmd_dir = Path(output_dir) / "lmd"
    lmd_file = lmd_dir / f"{video_name}.txt"

    # Skip if already processed
    if lmd_file.exists():
        existing = list(frame_dir.glob("*.jpg")) if frame_dir.exists() else []
        if len(existing) > 10:
            return "skipped"

    lmd_dir.mkdir(parents=True, exist_ok=True)

    # 1. Face crop detection (GPU lock only for detection)
    crop_info = get_stable_face_crop_fast(video_path, fa, gpu_lock)
    if crop_info is None:
        return "no_face"

    x1, y1, cw, ch = crop_info

    # 2. Extract frames (CPU — no lock)
    frame_paths = extract_frames_ffmpeg(
        video_path, frame_dir, target_fps, target_size, x1, y1, cw, ch
    )
    if len(frame_paths) < 10:
        return "too_few_frames"

    # 3. Landmark extraction (batched GPU)
    lmd_lines = extract_landmarks_batch(frame_paths, fa, target_size, gpu_lock)

    with open(lmd_file, "w") as f:
        f.write("\n".join(lmd_lines) + "\n")

    return f"ok ({len(frame_paths)})"


def gpu_worker(gpu_id, video_shard, output_dir, target_fps, target_size,
               threads_per_gpu, progress_dict):
    """One process per GPU, with multiple threads for CPU/IO overlap."""
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    import face_alignment
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device=device, flip_input=False
    )

    gpu_lock = threading.Lock()

    success, skipped, failed = 0, 0, 0
    done_count = 0

    def _process(vpath):
        nonlocal success, skipped, failed, done_count
        try:
            status = process_single_video(
                vpath, output_dir, target_fps, target_size, fa, gpu_lock
            )
            if status == "skipped":
                skipped += 1
            elif status.startswith("ok"):
                success += 1
        except Exception as e:
            if failed == 0:
                print(f"[Error] GPU {gpu_id} processing {vpath} failed: {e}")
            failed += 1

        done_count += 1
        progress_dict[gpu_id] = done_count

    with ThreadPoolExecutor(max_workers=threads_per_gpu) as executor:
        futures = [executor.submit(_process, vp) for vp in video_shard]
        for f in as_completed(futures):
            pass

    print(f"[GPU {gpu_id}] Done. Success={success} Skipped={skipped} Failed={failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_fps", type=float, default=25.0)
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--threads_per_gpu", type=int, default=6,
                        help="Threads per GPU for CPU/IO overlap (default: 6)")
    parser.add_argument("--max_videos", type=int, default=None)
    args = parser.parse_args()

    num_gpus = args.num_gpus or torch.cuda.device_count()
    total_workers = num_gpus * args.threads_per_gpu

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "lmd"), exist_ok=True)

    video_files = sorted(glob(os.path.join(args.input_dir, "*.mp4")))

    total = len(video_files)
    print(f"Videos: {total}")
    print(f"GPUs: {num_gpus} x {args.threads_per_gpu} threads = {total_workers} concurrent pipelines")
    print(f"Output: {args.output_dir}")
    print()

    # Split videos round-robin across GPUs
    shards = [[] for _ in range(num_gpus)]
    for i, vf in enumerate(video_files):
        shards[i % num_gpus].append(vf)

    manager = mp.Manager()
    progress_dict = manager.dict({i: 0 for i in range(num_gpus)})

    mp.set_start_method("spawn", force=True)
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, shards[gpu_id], args.output_dir,
                  args.target_fps, args.target_size,
                  args.threads_per_gpu, progress_dict)
        )
        p.start()
        processes.append(p)

    # Progress monitor
    pbar = tqdm(total=total, desc=f"All GPUs ({total_workers} workers)")
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
