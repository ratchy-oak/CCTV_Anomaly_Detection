import streamlit as st
from transformers import AutoTokenizer, AutoModelForVideoClassification
import torch

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("ratchy-oak/vivit-b-16x2-kinetics400-finetuned-cctv-surveillance")
model = AutoModelForVideoClassification.from_pretrained("ratchy-oak/vivit-b-16x2-kinetics400-finetuned-cctv-surveillance")

# Function to preprocess video frames
def preprocess_frame(frame):
    # Preprocess frame as needed (e.g., resize, normalize)
    return frame

# Function to perform inference on video frames
def predict_video(video_bytes):
    cap = cv2.VideoCapture(video_bytes)
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        preprocessed_frame = preprocess_frame(frame)

        # Convert to PyTorch tensor
        input_tensor = torch.tensor(preprocessed_frame).unsqueeze(0)

        # Perform inference
        outputs = model(input_tensor)

        # Process outputs as needed
        # (e.g., get predicted class probabilities)

    cap.release()

    return predictions

# Streamlit app
def main():
    st.title("Video Classification App")

    # File uploader
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

    if video_file:
        # Display video
        st.video(video_file)

        # Perform inference when video uploaded
        predictions = predict_video(video_file)

        # Display predictions
        st.write("Predictions:", predictions)

if __name__ == "__main__":
    main()
