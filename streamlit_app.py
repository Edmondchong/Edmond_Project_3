import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import requests
import numpy as np
import cv2


st.title("🧠 Edmond Chong's Brain MRI Tumor Classifier")
st.subheader("* Example MRI images are available on Github *")


# -------------------------------
# Model definition
# -------------------------------
class BrainTumorNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)


# -------------------------------
# Grad-CAM helper
# -------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def save_activation(module, input, output):
            self.activations = output

        target_layer.register_forward_hook(save_activation)
        target_layer.register_backward_hook(save_gradient)

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach().cpu().numpy()[0]

        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i].cpu().numpy()

        heatmap = np.mean(activations, axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max()

        return heatmap


# -------------------------------
# Load model
# -------------------------------
MODEL_PATH = Path("brain_tumor_resnet18.pth")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
        url = st.secrets["MODEL_URL"]
        r = requests.get(url, headers=headers)
        r.raise_for_status()  
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

    model = BrainTumorNet(num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()


# -------------------------------
# Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # ensure RGB
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])


# -------------------------------
# Streamlit UI
# -------------------------------
uploaded_file = st.file_uploader("Upload a Brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=200)

        if st.button("Predict Tumor Type"):
            with st.spinner("Predicting... ⏳"):
                img_tensor = transform(image).unsqueeze(0)

                # ---- Prediction ----
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred_class = torch.max(probs, 1)

                classes = ["glioma", "meningioma", "no tumor", "pituitary"]

                st.subheader("Prediction Result")
                st.write(f"Tumor Type: **{classes[pred_class.item()]}**")
                st.write(f"Confidence: **{conf.item():.2f}**")

                # ---- Grad-CAM ----
                gradcam = GradCAM(model, model.base_model.layer4[-1])  # last conv in ResNet18
                img_tensor.requires_grad = True
                heatmap = gradcam.generate(img_tensor, target_class=pred_class.item())

                # 🔹 Normalize heatmap properly
                heatmap = np.maximum(heatmap, 0)
                heatmap = heatmap / heatmap.max()
                                
                # 🔹 Invert values if important areas show up as blue
                heatmap = 1 - heatmap  

                # 🔹 Resize and apply colormap
                heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                overlay = cv2.addWeighted(np.array(image.convert("RGB")), 0.6, heatmap, 0.4, 0)

                st.subheader("Grad-CAM Heatmap")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original MRI", use_column_width=True)
                with col2:
                    st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)

    except Exception as e:
        st.error(f"Error processing image: {e}")
