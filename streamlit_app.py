import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api/inference/video")

st.title("Video Inference Demo")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
fps = st.number_input("Frames per second for inference", min_value=1, value=1)

if uploaded_file and st.button("Run Inference"):
    with st.spinner("Processing video..."):
        file_bytes = uploaded_file.read()
        files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
        data = {"fps": str(int(fps))}
        try:
            resp = requests.post(API_URL, files=files, data=data)
            resp.raise_for_status()
        except Exception as e:
            st.error(f"Request failed: {e}")
        else:
            results = resp.json().get("results", [])
            if results:
                st.success("Inference completed")
                df = pd.DataFrame(results)
                st.dataframe(df)
            else:
                st.info("No results received")
