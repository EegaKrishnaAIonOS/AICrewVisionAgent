"""Streamlit UI for uploading an image and running the CrewAI pipeline."""

import os
import tempfile
from pathlib import Path

import streamlit as st

from main2 import run_image_analysis

st.set_page_config(page_title="AI Crew Image Analyzer", layout="wide")
st.title("📸 AI Crew Image Query Assistant")
st.write(
    "Upload an image, ask any question about it, and let the CrewAI agents reason over the scene."
)
st.caption(
    "Tip: The first analysis of a new image consumes ~750 OpenAI tokens. "
    "Repeating questions on the same image is served from cache with zero extra vision tokens."
)


def _save_uploaded_file(uploaded_file) -> str:
    extension = Path(uploaded_file.name).suffix or ".png"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    temp_file.write(uploaded_file.getbuffer())
    temp_file.flush()
    temp_file.close()
    return temp_file.name


with st.form("analysis_form"):
    uploaded_image = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "gif", "webp"]
    )
    user_query = st.text_area(
        "Enter your question about the image",
        placeholder="e.g., What insights can you extract from this scene?",
    )
    submitted = st.form_submit_button("Run Analysis")

if submitted:
    if uploaded_image is None:
        st.error("Please upload an image before running the analysis.")
    elif not user_query.strip():
        st.error("Please enter a question to analyze.")
    else:
        temp_path = _save_uploaded_file(uploaded_image)
        st.image(uploaded_image, caption="Uploaded image", use_container_width=True)
        try:
            with st.spinner("Running CrewAI analysis..."):
                final_answer, task_outputs = run_image_analysis(temp_path, user_query)

            st.success("Analysis complete!")

            st.subheader("Final Answer")
            st.write(final_answer)

            # st.subheader("Agent Task Outputs")
            # for idx, (agent_name, raw_output) in enumerate(task_outputs, start=1):
            #     st.markdown(f"**Task {idx} – {agent_name}**")
            #     st.write(raw_output)
            #     st.divider()
        except Exception as exc:
            st.error(f"An error occurred while running the analysis: {exc}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
else:
    st.info("Upload an image and ask a question to get started.")

