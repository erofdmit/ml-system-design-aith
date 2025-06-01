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

def load_video(video_path: str) -> cv2.VideoCapture:
    """Load video from file"""
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise RuntimeError("Failed to open video file. Please check the path.")
    return video_capture

def draw_bounding_boxes(frame: np.ndarray, bounding_box_data: pd.DataFrame) -> np.ndarray:
    """Draw bounding boxes on the frame"""
    for index, row in bounding_box_data.iterrows():
        cv2.rectangle(frame, (int(row['x1']), int(row['y1'])), (int(row['x2']), int(row['y2'])), (255, 0, 0), 2)
    return frame

def display_frame(frame: np.ndarray, frame_data: pd.DataFrame) -> None:
    """Display the frame with bounding boxes"""
    class_name = frame_data.iloc[0]['class_name']
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Class: {class_name}")
    
def handle_user_input(frame_id: int, verified_data: pd.DataFrame, detection_data: pd.DataFrame) -> pd.DataFrame:
    """Handle user input for verification"""
    with st.form(f"verification_form_{frame_id}"):
        st.write(f"Frame ID: {frame_id}")
        classes = ['picket_post', 'kilometer_post']
        choice = st.radio("Choose class to keep:", classes)
        submitted = st.form_submit_button("Submit")
        if submitted:
            frame_data = detection_data[(detection_data['frame_id'] == frame_id)]
            frame_data['class_name'] = choice
            if not frame_data.empty:
                df = pd.read_csv(VERIFIED_PATH)
                df = pd.concat([df, frame_data], ignore_index=True)
                df.to_csv(VERIFIED_PATH, index=False)
            st.write("Verification submitted!")
            return verified_data
    return verified_data


def review_process(detection_data: pd.DataFrame, video_capture: cv2.VideoCapture, verified_data: pd.DataFrame) -> pd.DataFrame:
    """Main review process"""
    frame_id = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_data = detection_data[detection_data['frame_id'] == frame_id]
        if frame_data.shape[0] > 0:
            frame = draw_bounding_boxes(frame, frame_data)
            display_frame(frame, frame_data)
            verified_data = handle_user_input(frame_id, verified_data, detection_data)
            
        frame_id += 1
    video_capture.release()  # Release the video capture
    return verified_data


def save_verified_data(verified_data: pd.DataFrame) -> None:
    """Save verified data to CSV file"""
    verified_data.to_csv('verified_data.csv', index=False)
    st.success("Verified data saved successfully!")

def main() -> None:
    """Streamlit page configuration"""
    st.title("Bounding Box Verification Tool")
    st.write("Review each frame and validate the bounding box.")

    detection_data = load_detection_data(DATA_PATH)
    video_capture = load_video(VIDEO_PATH)
    
    if 'verified_data' not in st.session_state:
        st.session_state['verified_data'] = pd.DataFrame(columns=detection_data.columns)

    if st.button("Save Verified Data", key='save_verified_data'):
        save_verified_data(st.session_state['verified_data'])
        
    verified_data = st.session_state['verified_data']  # Retrieve verified data from session state
    verified_data = review_process(detection_data, video_capture, verified_data)  # Pass verified_data to review_process
    st.session_state['verified_data'] = verified_data
if __name__ == "__main__":
    main()