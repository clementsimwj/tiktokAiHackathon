import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification
from .inference import predict_video
from dotenv import load_dotenv
import os

"""
Test with your own video inputs. Add them to the videos folder!

"""
# ---------------------------
# 1️⃣ Setup device
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# 2️⃣ Reload processor & model
# ---------------------------
processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

model = AutoModelForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics"
)

# Load fine-tuned weights
state_dict = torch.load("ai_video_detector.pth", map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

# ---------------------------
# 3️⃣ Run inference on a test video
# ---------------------------
load_dotenv()
test_video_1 = os.getenv("test_video_1")
test_video_2 = os.getenv("test_video_2")
test_video_3 = os.getenv("test_video_3")
test_video_4 = os.getenv("test_video_4")
test_video_5 = os.getenv("test_video_5")
test_video_6 = os.getenv("test_video_6")

result_1 = predict_video(test_video_1, processor, model, device=device)
result_2 = predict_video(test_video_2, processor, model, device=device)
result_3 = predict_video(test_video_3, processor, model, device=device)
result_4 = predict_video(test_video_4, processor, model, device=device)
result_5 = predict_video(test_video_5, processor, model, device=device)
result_6 = predict_video(test_video_6, processor, model, device=device)

results = [result_1, result_2, result_3]
results = {os.getenv("test_video_name_1") : result_1,
           os.getenv("test_video_name_2") : result_2,
           os.getenv("test_video_name_3") : result_3,
           os.getenv("test_video_name_4") : result_4,
           os.getenv("test_video_name_5") : result_5,
           os.getenv("test_video_name_6"): result_6}

for key, value in results.items():
    if results[key] is not None:
        print(f"✅ {key}: {value}")
        print(f"   Real: {value['real']:.4f}")
        print(f"   AI:   {value['ai']:.4f}")
