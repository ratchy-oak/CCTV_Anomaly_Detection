import streamlit as st
import torch
from torchvision.io import read_video
from transformers import AutoTokenizer, AutoModelForVideoClassification

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("ratchy-oak/vivit-b-16x2-kinetics400-finetuned-cctv-surveillance")
model = AutoModelForVideoClassification.from_pretrained("ratchy-oak/vivit-b-16x2-kinetics400-finetuned-cctv-surveillance")

# Function to classify video
def classify_video(video_file):
    # Read video file
    video_tensor, audio_tensor, info = read_video(video_file)

    # Preprocess video frames (resizing, normalization, etc.)
    resized_frames = torch.nn.functional.interpolate(video_tensor, size=(224, 224))

    # Perform inference on each frame and aggregate results
    predictions = []
    for frame in resized_frames:
        input_tensor = frame.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        _, predicted_class = torch.max(probabilities, 1)
        predictions.append(predicted_class.item())

    # Aggregate predictions (taking the mode)
    final_prediction = max(set(predictions), key=predictions.count)
    
    return final_prediction

# Streamlit app
def main():
    st.title("Video Classification")

    # Upload video
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

    if uploaded_file is not None:
        # Display uploaded video
        st.video(uploaded_file)

        # Classify video when button is clicked
        if st.button("Classify"):
            prediction = classify_video(uploaded_file)
            st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
