import time
import streamlit as st
import numpy as np
from yolo_predctions import YOLO_Pred
from PIL import Image
from streamlit_webrtc import webrtc_streamer
import av

st.image('logo.png', width=250)
st.header('Bem-vindo ao nosso sistema!')
st.text('Faça o upload de uma foto para identificação automatizada de segurança!')

tabImg, tabVideo, tabWebCam = st.tabs(["Imagem", "Vídeo", "Webcam"])

with st.spinner('Carregando modelo...'):
  yolo=YOLO_Pred('Model/weights/best.onnx','data.yaml')
  

isSecurity = False
#Imagem
def upload_image():
  image_file = st.file_uploader(label='Upload Image')
  if image_file is not None:
      size_mb = image_file.size/(1024**2)
      details = {"filename":image_file.name,
                      "filetype":image_file.type,
                      "filesize": "{:,.2f} MB".format(size_mb)}
      if details["filetype"] in ('image/png', 'image/jpg', 'image/jpeg'):
          st.success('Imagem válida')
          return {"file":image_file,
                  "details":details}
      else:
          st.error('Imagem não suportada')
          return None
      
def SubmitImage(file):
  prediction = False
  image_obj = Image.open(file['file'])       
  
  col1 , col2 = st.columns(2)
  
  with col1:
      st.info('Preview of Image')
      st.image(image_obj)
      
  with col2:
      button = st.button('Get Detection from YOLO')
      if button:
          with st.spinner("""
          Detectando Objetos...
                          """):
              image_array = np.array(image_obj)
              pred_img = yolo.predictions(image_array)
              pred_img_obj = Image.fromarray(pred_img)
              prediction = True
          
  if prediction:
      st.subheader("Imagem com identificação")
      st.image(pred_img_obj)

#Video
def SubmitVideo(file):
  file_pred = yolo.predictions(file)
  video_bytes = file_pred.read()
  st.video(video_bytes)

def video_frame_callback(frame):
   img = frame.to_ndarray(format="bgr24")
   pred_img = yolo.predictions(img)
   return av.VideoFrame.from_ndarray(pred_img,format="bgr24")

def statusMessage(status):
  if status: 
    st.success('Todos procedimentos de segurança respeitados!', icon="✅")
  else:
    st.error('Procedimento de segurança burlado!! Atenção!', icon="🚨")

with tabImg:
  upload_file = upload_image()
  if upload_file:
    SubmitImage(upload_file)

with tabVideo:
  upload_file = st.file_uploader(label="Upload Video")
  if upload_file:
    SubmitVideo(upload_file)

with tabWebCam:
  webrtc_streamer(
     key='example',
     video_frame_callback=video_frame_callback,
     media_stream_constraints={'video': True,'audio': False}
     )

