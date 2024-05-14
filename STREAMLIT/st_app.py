import cv2
import numpy as np
import streamlit as st
from camera_input_live import camera_input_live
from PIL import Image

import torch
from torchvision import transforms


model_path = "resnet152unfreeze9 -30epoch.pt"
model = torch.load(model_path)
model.eval()

def predict_emotion(img, model):


    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img)
    predicted_emotion = torch.argmax(predictions, dim=1).item()
    return predicted_emotion

def get_emoji(emotion):
    emotion_to_emoji = {
        "Anger": "😠",
        "Contempt": "😒",
        "Disgust": "🤢",
        "Fear": "😨",
        "Happy": "😊",
        "Neutral": "😐",
        "Sad": "😢",
        "Surprise": "😲",
    }

    if emotion:
        return emotion_to_emoji.get(emotion, "🤔") + " " + emotion

st.title("Emotion Detection App")

st.write("This app detects your emotion from a picture of your face. 😊📸")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    bytes_data = uploaded_file.read()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    st.write("Detecting emotion...")

    emotion = predict_emotion(cv2_img, model)

    if emotion:
        emoji = get_emoji(emotion)
        st.write(f"Detected emotion: {emotion} {emoji}")
else:
    st.write("Please provide an image to detect the emotion.")
