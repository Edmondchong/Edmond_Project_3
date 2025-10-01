import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.title("🧠 Edmond Chong's Brain MRI Tumor Classifier")

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

@st.cache_resource
def load_model():
    model = BrainTumorNet(num_classes=4)
    model.load_state_dict(torch.load("brain_tumor_resnet18.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

uploaded_file = st.file_uploader("Upload a Brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=200)

        if st.button("Predict Tumor Type"):
            with st.spinner("Predicting... ⏳"):
                img_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred_class = torch.max(probs, 1)

                classes = ["glioma", "meningioma", "pituitary", "no tumor"]

                st.subheader("Prediction Result")
                st.write(f"Tumor Type: **{classes[pred_class.item()]}**")
                st.write(f"Confidence: **{conf.item():.2f}**")

    except Exception as e:
        st.error(f"Error processing image: {e}")
