import streamlit as st
import cv2
import os
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
#from PyPDF2 import PDF
#import pdfplumber

#@st.cache
   #def load_image(uploaded_img):
    #  img = Image.open(uploaded_img)
    #  return img

st.title('Defects Checker')
uploaded_img = st.file_uploader("Choose an image", ["jpg","jpeg","png"]) #image uploader, accept_multiple_files=True
#returned_pdf = #function_of_neural

if uploaded_img is not None:
   file_details={"filename":uploaded_img.name, "filetype":uploaded_img.type, "filesize":uploaded_img.size}
   st.write(file_details)
   with open (os.path.join("Result",uploaded_img.name),"wb") as f:
      f.write(uploaded_img.getbuffer())

#if returned_pdf is not None:
   #file_details={"filename":returned_pdf.name, "filetype":returned_pdf.type, "filesize":returned_pdf.size}
   #with open (os.path.join("Result",returned_pdf.name),"wb") as f:
      #f.write(returned_pdf.getbuffer())


