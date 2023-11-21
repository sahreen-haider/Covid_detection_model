import streamlit as st
import tensorflow as tf
import numpy as  np
import cv2
from PIL import Image
from flask import Flask, render_template
import os
import time


app = Flask(__name__)

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route("/streamlit")
def stream():
    st.title('Covid Detection')

    model = tf.keras.models.load_model('/Users/sahreenhaider/Documents/Covid_detection_model/model.h5')

    uploaded_image = st.file_uploader(':rainbow[Please Upload an image]: ', type=['jpg', 'jpeg', 'png'])
    if st.button('upload'):
        my_progress = st.progress(0.0)
        for _ in range(101):
            time.sleep(0.003)
            my_progress.progress(_ / 100.0)


    IMG_SIZE = (100, 100)

    if uploaded_image is not None:
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        image = cv2.resize(image, IMG_SIZE)
        image = image[:,:,:3]

        
        img_array = np.array(image)
        X = image/255
        X = X.reshape(1, 100, 100, 3)
        pred = model.predict(X)
        pred = np.argmax(pred, axis = 1)[0]


    if uploaded_image is not None:
        st.write('The algorithm Detects: ')
        if pred == 0:
            st.write('Covid')
        elif pred == 1:
            st.write('Healthy Person')
        elif pred == 2:
            st.write('**Some Other Disease**')

if __name__ == '__main__':
    app.run()
