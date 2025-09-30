# inference.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Model --------------------
class BrainTumorNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.base_model = models.resnet18(pretrained=False)  # pretrained=False in inference
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Load model
num_classes = 4
model = BrainTumorNet(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("brain_tumor_resnet18.pth", map_location=device))
model.eval()

# -------------------- Preprocessing --------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # match your training normalization
                         std=[0.5, 0.5, 0.5])
])

# Class labels
class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return {"tumor_type": class_names[pred.item()], "confidence": float(conf.item())}

# Quick test
if __name__ == "__main__":
    result = predict("test_mri.jpg")
    print(result)
