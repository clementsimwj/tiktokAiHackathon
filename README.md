# AI Video Detector

## Overview
This project is a machine learning-based system designed to detect whether an input video has been AI-generated or is real. By analyzing sampled frames from videos, the model classifies each video with probability scores indicating the likelihood of it being AI-generated or authentic.

## Features
- Extracts a fixed number of representative frames from any input video.
- Utilizes a pretrained video transformer model fine-tuned specifically for distinguishing AI-generated videos.
- Supports batch training and inference on local video datasets.
- Automatically handles corrupted or missing videos during data processing.
- Outputs interpretable probabilities for real vs. AI-generated content.
- Saves trained model checkpoints for reuse or further fine-tuning.

## Problem Statement
With the rise of high-quality AI-generated videos, it is increasingly difficult to verify content authenticity. This poses risks related to misinformation, fraud, and media manipulation. This tool provides automated detection to help users confidently identify AI-generated videos and maintain trust in video content.

## Development Tools
- Python
- PyTorch
- OpenCV
- PIL (Python Imaging Library)

## APIs
- Hugging Face Hub API for accessing pretrained models

## Libraries
- torch (PyTorch)
- torchvision (implied)
- transformers
- datasets (dependency)
- opencv-python
- Pillow
- tqdm
- huggingface-hub
- dotenv

## Assets
- Local datasets of labeled real and AI-generated videos used for training and evaluation

### Dataset Links
- AI-generated videos dataset: [svjack/Shoji_ai_videos](https://huggingface.co/datasets/svjack/Shoji_ai_videos)
- Real videos dataset: [hututao/video](https://huggingface.co/datasets/hututao/video)

### Additional Resources
- Supplementary project files and assets: [Google Drive Folder](https://drive.google.com/drive/folders/1T99PPDi_uDa6IkdgY0RL9-YCR_3jTlWO?usp=sharing)
- Move the file ai_video_detector.pth into the model folder and run app.py
