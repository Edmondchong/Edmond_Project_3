import streamlit as st
import requests

st.title("🧠 Edmond Chong's Brain MRI Tumor Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload a Brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded MRI Image", use_container_width=80)

    if st.button("Predict Tumor Type"):
        # Send file to FastAPI backend
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:8000/predict", files=files)

        if response.status_code == 200:
            result = response.json()
            st.subheader("Prediction Result")
            st.write(f"Tumor Type: **{result['tumor_type']}**")
            st.write(f"Confidence: **{result['confidence']:.2f}**")
        else:
            st.error("Error: Could not get prediction. Please check the API.")
