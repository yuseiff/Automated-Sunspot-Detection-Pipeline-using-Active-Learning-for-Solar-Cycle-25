import cv2
import os
import glob
from tqdm import tqdm
import subprocess

# ---------------- CONFIGURATION ---------------- #
year = "2025"
fps = 60

# We point to the year folder, not a specific month
base_dir = f"images/{year}"

# Update output filenames to be Year-based
raw_video = f"videos/{year}_FULL_raw.mp4"
compressed_video = f"videos/{year}_FULL_compressed.mp4"
# ----------------------------------------------- #

os.makedirs("videos", exist_ok=True)

print(f"Searching for images in {base_dir}...")

# 🔍 FIXED: Use recursive search ('**') to find ALL images in all subfolders (Months/Days)
# This will find images/2025/01/01/*.jpg, images/2025/07/15/*.jpg, etc.
image_files = sorted(glob.glob(os.path.join(base_dir, "**", "*.jpg"), recursive=True))

if not image_files:
    raise ValueError(f"No images found in {base_dir}. Check your folder structure.")

print(f"Found {len(image_files)} frames total for the year {year}")

# Read the first image to determine frame size
first = cv2.imread(image_files[0])
if first is None:
    raise ValueError(f"Could not read the first image: {image_files[0]}")

# Resize first frame to target resolution
# (Make sure to resize consistent with the loop below)
target_size = (1024, 1024)
first = cv2.resize(first, target_size)
h, w, _ = first.shape

# --- 1. Create Raw Video ---
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(raw_video, fourcc, fps, (w, h))

for img_path in tqdm(image_files, desc="Creating raw video"):
    frame = cv2.imread(img_path)
    # Check if image is valid
    if frame is None:
        print(f"Warning: Skipping corrupt file {img_path}")
        continue
        
    frame = cv2.resize(frame, target_size)
    video.write(frame)

video.release()
print("Raw video saved:", raw_video)

# --- 2. Compress with FFmpeg ---
print("Compressing with FFmpeg...")

try:
    subprocess.run([
        "ffmpeg",
        "-y",                 # Overwrite output without asking
        "-i", raw_video,
        "-vcodec", "libx264",
        "-crf", "24",         # Quality: 18 (high) to 28 (low)
        "-preset", "slow",    # Better compression
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        compressed_video
    ], check=True)
    print("Compressed video saved:", compressed_video)
    print("You can now delete the raw file if you want.")
    
except FileNotFoundError:
    print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in your system PATH.")
except subprocess.CalledProcessError as e:
    print(f"FFmpeg failed with error: {e}")