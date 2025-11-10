import streamlit as st
from core.pipeline import ScenePipeline

st.title("🧠 Scene Analyst — Vision-Language AI Agent (Offline Version)")

pipeline = ScenePipeline()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
question = st.text_input("Ask a question about the image:")

if uploaded_file and question:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    caption, answer, objects = pipeline.analyze("temp.jpg", question)
    st.image("temp.jpg", caption=caption)
    st.subheader("Detected Objects:")
    st.write(objects)
    st.subheader("Answer:")
    st.write(answer)
