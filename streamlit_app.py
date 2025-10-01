import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# App Title
# -----------------------------
st.title("🧠 Edmond Chong's Brain MRI Tumor Classifier")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 4)  # 4 classes
    model.load_state_dict(torch.load("brain_tumor_resnet18.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload a Brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    if st.button("Predict Tumor Type"):
        # Preprocess
        img_tensor = transform(image).unsqueeze(0)

        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_class = torch.max(probs, 1)

        # Classes (must match your training order)
        classes = ["glioma", "meningioma", "pituitary", "no tumor"]

        # Show result
        st.subheader("Prediction Result")
        st.write(f"Tumor Type: **{classes[pred_class.item()]}**")
        st.write(f"Confidence: **{conf.item():.2f}**")
