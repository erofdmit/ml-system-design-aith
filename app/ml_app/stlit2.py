import os
import pandas as pd
import cv2
import streamlit as st
import numpy as np

# Constants
MODEL_PATH = './app/ml_app'
DATA_PATH = os.path.join(MODEL_PATH, 'detection_results.csv')
VIDEO_PATH = os.path.join(MODEL_PATH, 'train-video.mp4')
VERIFIED_PATH = 'verified_data.csv'

def load_detection_data(data_path: str) -> pd.DataFrame:
    """Load detection data from CSV file"""
    detection_data = pd.read_csv(data_path)
    if detection_data.empty or not all(col in detection_data.columns for col in ['frame_id', 'x1', 'y1', 'x2', 'y2', 'class_name']):
        raise ValueError("Data is missing or incorrect. Please check your CSV file.")
    return detection_data

def draw_bounding_boxes(frame: np.ndarray, bounding_box_data: pd.DataFrame) -> np.ndarray:
    """Draw bounding boxes on the frame"""
    for index, row in bounding_box_data.iterrows():
        cv2.rectangle(frame, (int(row['x1']), int(row['y1'])), (int(row['x2']), int(row['y2'])), (255, 0, 0), 2)
    return frame

def display_frame(frame: np.ndarray, frame_data: pd.DataFrame) -> None:
    """Display the frame with bounding boxes"""
    class_name = frame_data.iloc[0]['class_name']
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Class: {class_name}")

def handle_user_input(frame_id: int, detection_data: pd.DataFrame) -> None:
    """Handle user input for verification"""
    with st.form(f"verification_form_{frame_id}"):
        st.write(f"Frame ID: {frame_id}")
        choice = st.radio("Verify", ("keep", "discard"))
        submitted = st.form_submit_button("Submit")
        if submitted:
            if choice == "keep":
                frame_data = detection_data[detection_data['frame_id'] == frame_id]
                if not frame_data.empty:
                    df = pd.read_csv(VERIFIED_PATH)
                    df = pd.concat([df, frame_data], ignore_index=True)
                    df.to_csv(VERIFIED_PATH, index=False)
            st.write("Verification submitted!")

def main() -> None:
    """Streamlit page configuration"""
    st.title("Bounding Box Verification Tool")
    st.write("Review each frame and validate the bounding box.")

    detection_data = load_detection_data(DATA_PATH)
    video_capture = cv2.VideoCapture(VIDEO_PATH)

    display_next_frame = True  # Flag to control whether to display the next frame

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_id = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        frame_data = detection_data[detection_data['frame_id'] == frame_id]

        if display_next_frame and not frame_data.empty:
            frame = draw_bounding_boxes(frame, frame_data)
            display_frame(frame, frame_data)
            handle_user_input(frame_id, detection_data)
            display_next_frame = False

        if not display_next_frame:
            st.empty()  # Clear the previous frame from the display
            display_next_frame = True  # Reset the flag for the next iteration

    video_capture.release()

    video_capture.release()

if __name__ == "__main__":
    main()
