import gradio as gr
import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification
from model.inference import predict_video

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor & model
processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

model = AutoModelForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics"
)

state_dict = torch.load("model/ai_video_detector.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

def process_video(video_path):
    if video_path is None:
        return "No video uploaded"

    result = predict_video(video_path, processor, model, device=device)

    if result is None:
        return "Could not process video."

    return (
        f"AI Probability: {result['ai']:.2%}\n"
        f"Real Probability: {result['real']:.2%}"
    )

iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload MP4"),
    outputs=gr.Textbox(label="Detection Result. NOTE THAT THIS MAY NOT BE ACCURATE AS IT IS ONLY A DEMO!"),
    title="AI Video Detector",
    description="Upload a video to analyze its whether it is AI."
)

iface.launch()
