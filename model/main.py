from huggingface_hub import login
import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoImageProcessor, AutoModelForVideoClassification

from .dataset_utils import LocalVideoDataset, collate_skip
from train import train_model
from dotenv import load_dotenv
import os

load_dotenv()
# ---------------------------
# 0️⃣ Hugging Face Token
# ---------------------------
TOKEN = token = os.getenv("TOKEN")
login(TOKEN)

# ---------------------------
# 1️⃣ Dataset Setup
# ---------------------------

import os
real_videos_root = os.getenv("real_videos_directory")
ai_videos_root = os.getenv("ai_videos_root")

# Dataset: AI videos labeled as 1, Real videos labeled as 0
real_dataset = LocalVideoDataset(real_videos_root, label=0, max_frames=16)
ai_dataset   = LocalVideoDataset(ai_videos_root, label=1, max_frames=16)


# Merge datasets
train_dataset = ConcatDataset([real_dataset, ai_dataset])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_skip)

print(f"✅ Loaded {len(real_dataset)} real + {len(ai_dataset)} AI videos = {len(train_dataset)} total")


# ---------------------------
# 2️⃣ Load Pretrained Model
# ---------------------------

# Load model directly
processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = AutoModelForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ---------------------------
# 3️⃣ Train Model
# ---------------------------
train_model(model, processor, train_loader, device, epochs=3, lr=1e-5)

# ---------------------------
# 4️⃣ Test Inference
# ---------------------------
# test_video = r"path\to\video.mp4"
# prob_ai = predict_video(test_video, feature_extractor, model, device)
# if prob_ai is not None:
#     print(f"Probability video is AI-generated: {prob_ai:.4f}")
