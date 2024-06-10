import gradio as gr
import streamlit as st

# Load model using Gradio
model = gr.load("models/ratchy-oak/vivit-b-16x2-kinetics400-finetuned-cctv-surveillance")

# Define function for video classification
def classify_video(video):
    prediction = model.predict(video)
    return prediction

# Define Streamlit app
def main():
    st.write("# Video Classification")
    st.write("Upload a video for classification:")

    # Define Gradio interface function
    def gr_interface():
        iface = gr.Interface(fn=classify_video, inputs="video", outputs="label")
        iface.launch()

    # Embed Gradio interface into Streamlit app
    gr_interface()

# Run the Streamlit app
if __name__ == "__main__":
    main()
