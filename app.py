import av
import time
import torch
import numpy as np
import streamlit as st

from transformers import VivitImageProcessor, VivitForVideoClassification

np.random.seed(0)

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


@st.cache_data
def load_model():
    image_processor = VivitImageProcessor.from_pretrained("ratchy-oak/vivit-b-16x2-kinetics400-finetuned-cctv-surveillance")
    model = VivitForVideoClassification.from_pretrained("ratchy-oak/vivit-b-16x2-kinetics400-finetuned-cctv-surveillance")
    return image_processor, model


def main():
    st.title("🕵️‍♂️ CCTV Anomaly Classification")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        container = av.open(uploaded_file)
        indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)

        image_processor, model = load_model()

        inputs = image_processor(list(video), return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predicted_label = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_label]

        normal_text = """
        This class likely represents situations or incidents that are considered typical
        or non-threatening. It could be used as a baseline for comparison against other
        classes to identify abnormal or concerning events.
        """

        abuse_text = """
        This class likely pertains to instances of physical, emotional, or other forms
        of abuse. It could include domestic violence, child abuse, or any other form of
        mistreatment towards individuals.
        """

        arson_text = """
        Arson refers to the intentional setting of fires to property or land.
        It's a criminal act and can lead to property damage, injury, or loss of life.
        """

        burglary_text = """
        Burglary involves the unlawful entry into a building or property with the intent
        to commit theft, vandalism, or other crimes. It differs from robbery in that it
        typically occurs when the building is unoccupied.
        """

        explosion_text = """
        This class likely refers to incidents involving the sudden release of energy
        in the form of a violent expansion. Explosions can occur due to various reasons,
        including accidents, industrial mishaps, or deliberate acts.
        """

        roadaccidents_text = """
        This class encompasses incidents involving vehicles on roads, including
        car accidents, collisions, and other traffic-related incidents.
        It's a broad category that can involve various levels of severity.
        """

        shooting_text = """
        This class refers to incidents where firearms are discharged, potentially
        resulting in injury or death. Shootings can occur in various contexts,
        including criminal activity, self-defense situations, or accidents.
        """
        
        def stream_text(text):
            for word in text.split(" "):
                yield word + " "
                time.sleep(0.02)

        if (predicted_class == "normal"):
            if st.button(predicted_class):
                st.write_stream(stream_text(normal_text))
        elif (predicted_class == "abuse"):
            if st.button(predicted_class):
                st.write_stream(stream_text(abuse_text))
        elif (predicted_class == "arson"):
            if st.button(predicted_class):
                st.write_stream(stream_text(arson_text))
        elif (predicted_class == "burglary"):
            if st.button(predicted_class):
                st.write_stream(stream_text(burglary_text))
        elif (predicted_class == "explosion"):
            if st.button(predicted_class):
                st.write_stream(stream_text(explosion_text))
        elif (predicted_class == "roadaccidents"):
            if st.button(predicted_class):
                st.write_stream(stream_text(roadaccidents_text))
        elif (predicted_class == "shooting"):
            if st.button(predicted_class):
                st.write_stream(stream_text(shooting_text))

if __name__ == "__main__":
    main()
