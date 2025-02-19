import glob
import streamlit as st
from PIL import Image
import cv2
import os
import time
import torch
import ultralytics
from ultralytics import YOLO
from collections import Counter
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide")

cfg_model_path = 'models/best 908.pt'
model = None
confidence = .25

def image_input(data_src):
    img_file = None
    if data_src == 'Sample data':
        # get all sample images
        img_path = glob.glob('data/sample_images/*')
        img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            # Create timestamp-based filename to prevent overwriting
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = img_bytes.name.split('.')[-1]
            img_file = f"data/uploaded_data/{timestamp}_upload.{file_extension}"
            
            # Save image to the folder
            os.makedirs("data/uploaded_data", exist_ok=True)
            Image.open(img_bytes).save(img_file)
            st.sidebar.success(f"Image saved to {img_file}")

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            img, result = infer_image(Image.open(img_file))
            counter = quantity_estimate(result)
            st.image(img, caption="Model prediction")
            for k, v in counter.items():
                st.write(f"{v} {result[0].names[k]}")
            
            # Save the prediction result
            save_path = f"data/results/{os.path.basename(img_file)}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path)
            st.success(f"Prediction saved to {save_path}")

def video_input(data_src):
    vid_file = None
    if data_src == 'Sample data':
        vid_file = "data/sample_videos/sample.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi', 'mkv'])
        if vid_bytes:
            # Create timestamp-based filename to prevent overwriting
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = vid_bytes.name.split('.')[-1]
            vid_file = f"data/uploaded_data/{timestamp}_upload.{file_extension}"
            
            # Save video to folder
            os.makedirs("data/uploaded_data", exist_ok=True)
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())
            st.sidebar.success(f"Video saved to {vid_file}")

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        # Setup for saving processed video
        save_output = st.sidebar.checkbox("Save processed video", value=True)
        output_path = None
        out = None
        
        if save_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/results/{timestamp}_processed_video.mp4"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        stop = st.button("Stop")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Video processing complete.")
                break
            frame = cv2.resize(frame, (width, height))
            output_img, _ = infer_image(frame)
            
            # Convert PIL image back to OpenCV format for saving
            if save_output and out is not None:
                cv_output = np.array(output_img)
                cv_output = cv_output[:, :, ::-1]  # RGB to BGR
                out.write(cv_output)
            
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")
            if stop:
                if save_output and out is not None:
                    out.release()
                    st.success(f"Processed video saved to {output_path}")
                cap.release()
                break

        if save_output and out is not None:
            out.release()
            st.success(f"Processed video saved to {output_path}")
        cap.release()

def live_input():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam. Please check your camera connection.")
            return
            
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        # Option to save snapshots
        save_snapshot = st.sidebar.button("Take Snapshot")
        
        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        
        # The stop button is intentionally removed to keep camera always on
        # Live feed runs until app is closed
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to get frame from camera. Camera may be disconnected.")
                break
                
            frame = cv2.resize(frame, (width, height))
            output_img, results = infer_image(frame)
            
            # Take snapshot if button pressed
            if save_snapshot:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_path = f"data/snapshots/{timestamp}_live_snapshot.jpg"
                os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
                
                # Save both original and predicted image
                cv2.imwrite(f"data/snapshots/{timestamp}_original.jpg", frame)
                np_img = np.array(output_img)
                cv2.imwrite(snapshot_path, np_img[:, :, ::-1])  # RGB to BGR conversion
                
                st.sidebar.success(f"Snapshot saved to {snapshot_path}")
                save_snapshot = False  # Reset button state
            
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")
            
            # Small sleep to prevent excessive CPU usage
            time.sleep(0.01)
    except Exception as e:
        st.error(f"Error in live feed: {str(e)}")
    finally:
        if 'cap' in locals() and cap is not None and cap.isOpened():
            cap.release()

def infer_image(frame, size=None):
    results = model.predict(source=frame, show=False, conf=confidence, save=False)
    for r in results:
        im_array = r.plot()  
        im = Image.fromarray(im_array[..., ::-1])  
    return im, results

def quantity_estimate(result):    
    counter = Counter(result[0].boxes.cls.numpy().astype(int))
    return counter

@st.cache_resource
def load_model(cfg_model_path, device):
    # Override ultralytics' torch_safe_load to use weights_only=False
    def custom_torch_safe_load(file):
        return torch.load(file, map_location="cpu", weights_only=False), file
    
    # Temporarily replace the function
    original_load = ultralytics.nn.tasks.torch_safe_load
    ultralytics.nn.tasks.torch_safe_load = custom_torch_safe_load
    
    try:
        model_ = YOLO(cfg_model_path)
        model_.to(device)
        print("Model loaded successfully")
        return model_
    finally:
        # Restore original function
        ultralytics.nn.tasks.torch_safe_load = original_load

def main(): 
    # global variables
    global model, confidence, cfg_model_path

    st.title("Object Recognition Dashboard")
    st.sidebar.title("Settings")

    # Create necessary directories
    folders = [
        "data/uploaded_data", 
        "data/sample_images", 
        "data/sample_videos", 
        "data/results",
        "data/snapshots",
        "models"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # check if model file is available
    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!, please add to the model folder.", icon="⚠️")
    else:
        # device options
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
        else:
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

        # load model
        try:
            model = load_model(cfg_model_path, device_option)

            # confidence slider
            confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

            model.classes = list(model.names.keys())

            st.sidebar.markdown("---")

            # input options
            input_option = st.sidebar.radio("Select input type: ", ['image', 'video', 'live'])

            # input src option
            if input_option != 'live':
                data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])

            if input_option == 'image':
                image_input(data_src)
            elif input_option == 'video':
                video_input(data_src)
            else:
                live_input()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.error("Please check if the model file is correctly formatted and accessible.")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass