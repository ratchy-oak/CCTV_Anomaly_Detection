import streamlit as st
from transformers import AutoTokenizer, AutoModelForVideoClassification
import cv2  # For video processing

def load_model_and_tokenizer():
    """Loads the video classification model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("ratchy-oak/vivit-b-16x2-kinetics400-finetuned-cctv-surveillance")
        model = AutoModelForVideoClassification.from_pretrained("ratchy-oak/vivit-b-16x2-kinetics400-finetuned-cctv-surveillance")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model and tokenizer: {e}")
        return None, None  # Handle errors gracefully

tokenizer, model = load_model_and_tokenizer()  # Load on module import

def classify_video(video_file):
    if tokenizer is None or model is None:
        st.error("Error loading model. Please check console for details.")
        return None, None

    # Read the video using OpenCV
    cap = cv2.VideoCapture(video_file.name)

    predictions = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame (NumPy array) to bytes (common approach)
        _, encoded_frame = cv2.imencode('.jpg', frame)
        frame_bytes = encoded_frame.tobytes()

        # Process individual frame using the pipeline
        prediction = model([frame_bytes], return_tensors="pt")  # Assuming model takes byte tensors
        predicted_class = prediction.logits.argmax(-1).item()  # Extract predicted class

        predictions.append({"class": predicted_class, "score": prediction.logits[0][predicted_class].item()})  # Store class and confidence

    cap.release()

    # Choose the top prediction (or modify for multiple predictions)
    top_prediction = max(predictions, key=lambda x: x["score"])["class"]
    confidence = max(predictions, key=lambda x: x["score"])["score"]

    return top_prediction, confidence

def main():
    st.title("CCTV Anomaly Classfication")

    uploaded_file = st.file_uploader("Choose a video file:", type=["mp4", "avi"])

    if uploaded_file is not None:
        prediction, confidence = classify_video(uploaded_file)
        if prediction is not None:
            st.success(f"Predicted Class: {prediction}")
            st.write(f"Confidence Score: {confidence:.2f}")  # Format confidence to 2 decimal places

if __name__ == "__main__":
    main()
