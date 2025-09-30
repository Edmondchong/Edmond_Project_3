import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definition ---
class BrainTumorNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load model
model = BrainTumorNet(num_classes=4).to(device)
model.load_state_dict(torch.load("brain_tumor_resnet18.pth", map_location=device))
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# --- Streamlit App ---
st.title("🧠 Edmond Chong's Brain MRI Tumor Classifier")

uploaded_file = st.file_uploader("Upload a Brain MRI image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)

    labels = ["glioma", "meningioma", "pituitary", "no tumor"]
    st.success(f"Prediction: **{labels[pred.item()]}**")
