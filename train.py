import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from PIL import Image

def train_model(model, processor, train_loader, device, epochs=3, lr=1e-5):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        count = 0

        for batch in train_loader:
            if batch is None:
                continue
            frames_batch, labels_batch = batch  # lists

            # Each video in batch
            for frames, label in zip(frames_batch, labels_batch):
                # Convert numpy frames → PIL
                pil_frames = [Image.fromarray(frame) for frame in frames]

                inputs = processor(pil_frames, return_tensors="pt")["pixel_values"].to(device)
                labels = torch.tensor([label]).to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                logits = outputs.logits
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / max(count, 1)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "ai_video_detector.pth")
    print("✅ Model saved as ai_video_detector.pth")
