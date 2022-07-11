
import streamlit as st
import glob
#import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
from numpy.linalg import norm
#from google.colab.patches import cv2_imshow
from PIL import Image
import sklearn 

#Resnet Model building
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
model = ResNet50(include_top = False,weights= 'imagenet' )
model.trainable = False


#CNN Model building
import keras
from keras.layers import GlobalMaxPooling2D
model = keras.Sequential(
  [model,
  GlobalMaxPooling2D()]
)


#this list contains the 2048 features for every training images
images_list = pickle.load(open('/app/celebslookalike/images_list.pkl', 'rb'))
name = pickle.load(open('/app/celebslookalike/name.pkl', 'rb'))
features_list2 = pickle.load(open('/app/celebslookalike/features_list2.pkl', 'rb'))


#Nearaest Neigbors modeel building to predict our neighbors
from sklearn.neighbors import NearestNeighbors
NNmodel = NearestNeighbors(n_neighbors = 1, metric= 'euclidean', algorithm='brute') 
NNmodel.fit(features_list2)

#finding nearest neighbors
def NearestNeighbors_func(predicting_img_features):
  return NNmodel.neighbors([predicting_img_features])


#feature extraction for uploaded images
def feature_extraction(uploaded_img):
  uploaded_img = np.expand_dims(uploaded_img, axis=0)
  uploaded_img = preprocess_input(uploaded_img)
  result = model.predict(uploaded_img)
  result = result.flatten()
  result = result/norm(result)
  return result




st.title('Which Celebrity Do You Look Alike ? ')
st.subheader('Use our Face Recognition model to find your celebrity look alike')
uploaded_file = st.file_uploader('Upload your photo to find : ')
if uploaded_file is not None:
  st.image(uploaded_file)
  uploaded_file = Image.open(uploaded_file)
  uploaded_file = uploaded_file.resize((224,224))
  uploaded_file = np.array(uploaded_file)


  extracted_features = feature_extraction(uploaded_file)
  neighbors_dist,indices = NNmodel.kneighbors([extracted_features])
  indices = indices.flatten().tolist()
  accur = neighbors_dist.flatten().tolist()
  accur = 100 - accur[0]

  img = images_list[indices[0]]
  img = cv2.imread(img)
  st.write('Your look alike is : ', name[indices[0]])
  st.write('Features matched with accuracy of ',accur, '%')
  st.image(img)
  st.balloons()
  
