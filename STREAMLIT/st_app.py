import cv2
import numpy as np
import streamlit as st
from camera_input_live import camera_input_live
from PIL import Image


def predict_emotion(img):
    """
    Load model and predict here
    """
    pass


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

image = camera_input_live(show_controls=False)

if image is not None:
    st.image(image)
    bytes_data = image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    st.write("Detecting emotion...")

    emotion = predict_emotion(cv2_img)

    if emotion:
        emoji = get_emoji(emotion)
        st.write(f"Detected emotion: {emotion} {emoji}")
else:
    st.write("Please provide an image to detect the emotion.")
