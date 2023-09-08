from pydoc import classname
from pyexpat import model
from re import S
from typing import List

import cv2
import os
import torch
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.colors as mcolors
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

from config import CLASSES, WEBRTC_CLIENT_SETTINGS

confidence = 0.25

st.set_page_config(
    page_title="Kursi-Check",
)

st.title("Deteksi Kursi Kosong Dan Terisi")


#region Functions
# --------------------------------------------

@st.cache
def load_model(model_type='s'):
    return torch.hub.load('ultralytics/yolov5', 
                          'yolov5{}'.format(model_type), 
                          pretrained=True
                          )
model = load_model('s')
@st.cache
def get_preds(img : np.ndarray) -> np.ndarray:
   
    return model([img]).xyxy[0].numpy()

def process_video(vid_bytes, confidence):
    vid_file = "data/video_upload/video_upload." + vid_bytes.name.split('.')[-1]
    with open(vid_file, 'wb') as out:
        out.write(vid_bytes.read())
    st.sidebar.video(vid_bytes)
    if not os.path.exists('videos'):
        os.makedirs('videos')
    with open(os.path.join('videos', vid_bytes.name), 'wb') as f:
        f.write(vid_bytes.getbuffer())

    st1, st2 = st.columns([2, 1])
    with st1:
        st.markdown("## Kursi Kosong")
        st1_Kursi_Kosong_count = st.markdown("__")
    with st2:
        st.markdown("## Kursi Terisi")
        st2_Kursi_Terisi_count = st.markdown("__")

    output = st.empty()
    prev_time = 0
    curr_time = 0
    video_started = False
    cap = cv2.VideoCapture(vid_file)

    Kursi_Terisi_count = 0  # Initialize counters
    Kursi_Kosong_count = 0

    if st.sidebar.button("Mulai"):
        video_started = True

    while video_started:
        ret, frame = cap.read()
        if not ret:
            st.write("Tidak dapat membaca frame, akhir stream? Keluar ....")
            break

        result = model([frame]).xyxy[0].numpy()
        result = result[np.isin(result[:, -1], target_class_ids)]
        
        # Filter hasil deteksi berdasarkan confidence
        result = result[result[:, 4] >= confidence]  # Hanya objek dengan confidence lebih besar atau sama dengan nilai confidence yang ditentukan

        for bbox_data in result:
            xmin, ymin, xmax, ymax, conf, label = bbox_data
            p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
            if CLASSES[label] == 'Kursi_terisi':
                color = (0, 0, 255)  # Merah untuk Kursi Terisi
            elif CLASSES[label] == 'Kursi_kosong':
                color = (0, 255, 0)  # Hijau untuk Kursi Kosong

            frame = cv2.rectangle(frame, p0, p1, color, 2)
            frame = cv2.putText(frame, f"{conf:.2f}", (int(xmin), int(ymin) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        Kursi_Terisi_count = sum(1 for bbox_data in result if bbox_data[-1] == CLASSES.index('Kursi_terisi'))
        Kursi_Kosong_count = sum(1 for bbox_data in result if bbox_data[-1] == CLASSES.index('Kursi_kosong'))

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        st1_Kursi_Kosong_count.markdown(f"**{Kursi_Kosong_count}**")
        st2_Kursi_Terisi_count.markdown(f"**{Kursi_Terisi_count}**")

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output.image(frame)

    cap.release()

def get_colors(indexes : List[int]) -> dict:
    to_255 = lambda c: int(c*255)
    tab_colors = list(mcolors.TABLEAU_COLORS.values())
    tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) 
                                                for name_color in tab_colors]
    base_colors = list(mcolors.BASE_COLORS.values())
    base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
    rgb_colors = tab_colors + base_colors
    rgb_colors = rgb_colors*5

    color_dict = {}
    for i, index in enumerate(indexes):
        if i < len(rgb_colors):
            color_dict[index] = rgb_colors[i]
        else:
            color_dict[index] = (255,0,0)

    return color_dict

def get_legend_color(class_name : int):
    index = CLASSES.index(class_name)
    color = rgb_colors[index]
    return 'background-color: rgb({color[0]},{color[1]},{color[2]})'.format(color=color)


#endregion

# UI elements
# ----------------------------------------------------

#sidebar
confidence = st.sidebar.slider('Kepercayaan', min_value=0.1, max_value=1.0, value=0.50)

st.sidebar.markdown("---")

# Pilihan objek yang ingin dideteksi
st.sidebar.markdown("## Objek yang ingin dideteksi")
classes_selector = st.sidebar.multiselect('Pilih Objek', 
                                        CLASSES, default='Kursi_kosong')

st.sidebar.markdown("---")

prediction_mode = st.sidebar.selectbox(
    "",
    ('Video', 'Real-Time'),
    index=1,
    format_func=lambda mode: mode)


# Prediction section
# ---------------------------------------------------------

#target labels and their colors
#target_class_ids 
if classes_selector:
    target_class_ids = [CLASSES.index(class_name) for class_name in classes_selector]
else:
    target_class_ids = [0]

rgb_colors = get_colors(target_class_ids)
detected_ids = None


if prediction_mode == 'Video':
    vid_bytes = st.sidebar.file_uploader("Unggah video", type=['mp4', 'mpv', 'avi'])
    if vid_bytes:
        process_video(vid_bytes, confidence)
    
elif prediction_mode == 'Real-Time':
    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.model = model
            self.rgb_colors = rgb_colors
            self.target_class_ids = target_class_ids
            self.confidence = confidence  # Tambahkan parameter confidence

        def get_preds(self, img : np.ndarray) -> np.ndarray:
            return self.model([img]).xyxy[0].numpy()

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            result = self.get_preds(img)
            result = result[np.isin(result[:,-1], self.target_class_ids)]
            
            # Filter hasil deteksi berdasarkan confidence
            result = result[result[:, 4] >= self.confidence]  # Hanya objek dengan confidence lebih besar atau sama dengan nilai confidence yang ditentukan
                    
            for bbox_data in result:
                xmin, ymin, xmax, ymax, conf, label = bbox_data
                p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
                # Tentukan warna berdasarkan jenis objek yang terdeteksi
                if CLASSES[label] == 'Kursi_terisi':
                    color = (255, 0, 0)  # Merah untuk Kursi Terisi
                elif CLASSES[label] == 'Kursi_kosong':
                    color = (0, 255, 0)  # Hijau untuk Kursi Kosong

                # Gambar bounding box dengan warna yang sesuai
                img = cv2.rectangle(img, p0, p1, color, 2)
                label_text = f"{conf:.2f}"
                img = cv2.putText(img, label_text, (int(xmin), int(ymin) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
            
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    ctx = webrtc_streamer(
        key="example", 
        video_transformer_factory=VideoTransformer,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        )

    if ctx.video_transformer:
        ctx.video_transformer.model = model
        ctx.video_transformer.rgb_colors = rgb_colors
        ctx.video_transformer.target_class_ids = target_class_ids
