import streamlit as st
import os
from sklearn.neighbors import NearestNeighbors
import tensorflow
from tensorflow.keras.preprocessing import  image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import cv2
import pickle


# Define model
def get_model():
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False

    model = tensorflow.keras.Sequential([
        model,
        tensorflow.keras.layers.GlobalAveragePooling2D(),
    ])
    return model

# Extract features
def extract_features(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_arr = image.img_to_array(img)
    expanded_img_arr = np.expand_dims(img_arr, axis=0)
    preprocess_img = preprocess_input(expanded_img_arr)
    result = model.predict(preprocess_img, verbose=0).flatten()
    normalized_result = result / np.linalg.norm(result)
    return normalized_result

# Get inference
def get_inference(image_path,model):
    normalized_result = extract_features(image_path, model)
    neighbours = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbours.fit(feature_list)
    distances, indices = neighbours.kneighbors([normalized_result])
    return indices[0], distances[0]  


# Upload a file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0
    
######## MAIN ########

# Title
st.title('Fashion Recommendation System')

feature_list = pickle.load(open("feature_list.pkl", "rb"))
filenames = pickle.load(open("filenames.pkl", "rb"))

uploaded_file = st.file_uploader("Choose an image...")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the file first
        st.image(Image.open(uploaded_file))
        # Get the inference
        model = get_model()
        indices, distances = get_inference(os.path.join('uploads',uploaded_file.name), model)
        # Display the recommendations
        st.subheader('Similar Products')
        cols = st.columns(5)
        for i in range(len(indices)):
            img = Image.open(filenames[indices[i]])
            resized_img = img.resize((200,300))
            cols[i].image(resized_img, use_column_width=True)
    else:
        st.warning('Some error occured in file upload. Please try again!')
    
