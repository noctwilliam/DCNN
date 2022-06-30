# Tensorflow / Keras
import tensorflow as tf # used to access argmax function
from tensorflow import keras # for building Neural Networks
print('Tensorflow/Keras: %s' % keras.__version__) # print version
from keras.models import Sequential, load_model # for creating a linear stack of layers for our Neural Network
from keras import Input # for instantiating a keras tensor
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout # for adding Concolutional and densely-connected NN layers.


# Data manipulation
import pandas as pd # for data manipulation
print('pandas: %s' % pd.__version__) # print version
import numpy as np # for data manipulation
print('numpy: %s' % np.__version__) # print version

# Sklearn
import sklearn # for model evaluation
print('sklearn: %s' % sklearn.__version__) # print version
from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn.preprocessing import OrdinalEncoder # for encoding labels

# Visualization
import cv2 # for ingesting images
print('OpenCV: %s' % cv2.__version__) # print version
import matplotlib 
import matplotlib.pyplot as plt # for showing images
print('matplotlib: %s' % matplotlib.__version__) # print version

#GUI
import tkinter as tk
from tkinter import filedialog
from turtle import title
import gradio as gr

# Other utilities
import sys
import os

enc = OrdinalEncoder()
loaded_model = tf.keras.models.load_model('my_model.h5')

def predict_img(flower_img):
    img = cv2.resize(flower_img, (128,128))
    img = img / 255.0
    img = img[np.newaxis, ...]
    prediction_probabilities = loaded_model.predict(img)
    return {enc.categories_[0][i] : float(prediction_probabilities[0][i]) for i in range(5)}

demo = gr.Interface(predict_img, 
            inputs=gr.inputs.Image(shape=(224,224)),
            outputs=gr.outputs.Label(num_top_classes=5),
            interpretation="default")

demo.launch(debug=TRUE)