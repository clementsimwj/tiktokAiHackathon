import torch
from dataset_utils import extract_frames
from PIL import Image

def predict_video(video_path, processor, model, device="cpu", max_frames=16):
    frames = extract_frames(video_path, max_frames=max_frames)
    if frames is None:
        print(f"⚠️ Skipping corrupted/missing video: {video_path}")
        return None

    # ✅ Convert numpy frames → PIL.Image
    pil_frames = []
    for f in frames:
        if f.ndim == 4:  # sometimes shape is (1, H, W, C)
            f = f.squeeze(0)
        pil_frames.append(Image.fromarray(f))

    # ✅ Pass list of PIL frames directly
    inputs = processor(pil_frames, return_tensors="pt")["pixel_values"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        prob_ai = probs[0, 1].item()  # class 1 = AI
        prob_real = probs[0, 0].item()

    return {"real": prob_real, "ai": prob_ai}

