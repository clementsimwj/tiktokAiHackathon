import os
import cv2
import numpy as np
from torch.utils.data import Dataset

def extract_frames(video_path, target_size=(224, 224), max_frames=16):
    """
    Extract exactly max_frames frames from a video.
    Pads by repeating last frame if video is too short.
    """
    if not os.path.exists(video_path):
        print(f"⚠️ File not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Could not open video: {video_path}")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        print(f"⚠️ No frames extracted: {video_path}")
        return None

    # Sample exactly max_frames frames
    if len(frames) >= max_frames:
        idxs = np.linspace(0, len(frames)-1, max_frames, dtype=int)
        frames = [frames[i] for i in idxs]
    else:
        while len(frames) < max_frames:
            frames.append(frames[-1])

    return np.array(frames)  # [T, H, W, 3]


class LocalVideoDataset(Dataset):
    """
    Dataset for local videos in a folder (nested allowed).
    Assigns label provided (0 = real, 1 = AI).
    """
    def __init__(self, videos_root, label, target_size=(224, 224), max_frames=16):
        self.video_paths = []
        for root, _, files in os.walk(videos_root):
            for f in files:
                if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                    self.video_paths.append(os.path.join(root, f))
        self.label = label
        self.target_size = target_size
        self.max_frames = max_frames

        print(f"✅ Loaded {len(self.video_paths)} videos from folder: {videos_root}")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        frames = extract_frames(path, self.target_size, self.max_frames)
        if frames is None:
            return None
        return frames, self.label


def collate_skip(batch):
    """
    Collate function that skips corrupted videos.
    """
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    frames, labels = zip(*batch)
    return list(frames), list(labels)
