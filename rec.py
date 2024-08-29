import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os
from numpy.linalg import norm
import streamlit as st
from PIL import Image
def run():
    st.header('Fashion Recommendation System')

    # Load precomputed image features and filenames
    image_features = pkl.load(open("C:/Users/Lenovo/OneDrive/Desktop/mini proj/embeddings.pkl", 'rb'))
    filenames = pkl.load(open("C:/Users/Lenovo/OneDrive/Desktop/mini proj/filenames.pkl", 'rb'))

    # Load descriptions from the cleaned CSV
    descriptions_df = pd.read_csv("C:/Users/Lenovo/OneDrive/Desktop/mini proj/styles_c.csv")

    # Verify that all required columns exist in the CSV
    required_columns = ['productDisplayName', 'Price', 'Rating', 'usage', 'subCategory']
    if not all(column in descriptions_df.columns for column in required_columns):
        missing = [column for column in required_columns if column not in descriptions_df.columns]
        st.error(f"The following columns do not exist in the CSV file: {', '.join(missing)}")
    else:
        descriptions_df['id'] = descriptions_df['id'].astype(str)

        # Create a dictionary to map filenames to descriptions with all necessary details
        descriptions = {}
        for index, row in descriptions_df.iterrows():
            descriptions[row['id'] + '.jpg'] = {
                "productDisplayName": row['productDisplayName'],
                "Price": row['Price'],
                "Rating": row['Rating'],
                "usage": row['usage'],
                "subCategory": row['subCategory']
            }

        # Function to extract features from an image
        def extract_features_from_images(image_path, model):
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_expand_dim = np.expand_dims(img_array, axis=0)
            img_preprocess = preprocess_input(img_expand_dim)
            result = model.predict(img_preprocess).flatten()
            norm_result = result / norm(result)
            return norm_result

        # Initialize the model
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        model = tf.keras.models.Sequential([base_model, GlobalMaxPool2D()])

        # Initialize NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(image_features)

        # List of image paths and corresponding items to be displayed
        image_data = [
            {"path": "C:/Users/Lenovo/OneDrive/Desktop/mini proj/images/1163.jpg", "item": "Jersey"},
            {"path": "C:/Users/Lenovo/OneDrive/Desktop/mini proj/images/2093.jpg", "item": "Shirt"},
            {"path": "C:/Users/Lenovo/OneDrive/Desktop/mini proj/images/1529.jpg", "item": "T-Shirt"},
            {"path": "C:/Users/Lenovo/OneDrive/Desktop/mini proj/images/1525.jpg", "item": "Bags"},
            {"path": "C:/Users/Lenovo/OneDrive/Desktop/mini proj/images/1561.jpg", "item": "Women T-Shirt"},
            {"path": "C:/Users/Lenovo/OneDrive/Desktop/mini proj/images/1597.jpg", "item": "Handbag"},
            {"path": "C:/Users/Lenovo/OneDrive/Desktop/mini proj/images/5071.jpg", "item": "Watches"},
            {"path": "C:/Users/Lenovo/OneDrive/Desktop/mini proj/images/38216.jpg", "item": "Sunglasses"}
        ]

        # Display images and buttons in a grid view
        num_columns = 4
        rows = len(image_data) // num_columns + (len(image_data) % num_columns > 0)
        
        recommended_images = []  # List to store recommended images

        for row in range(rows):
            cols = st.columns(num_columns)
            for col_index in range(num_columns):
                img_index = row * num_columns + col_index
                if img_index < len(image_data):
                    img_data = image_data[img_index]
                    with cols[col_index]:
                        st.write(f"**{img_data['item']}**")
                        img = Image.open(img_data['path'])
                        filename = os.path.basename(img_data['path'])
                        description_data = descriptions.get(filename, {"productDisplayName": "No description available"})
                        st.image(img, caption=description_data['productDisplayName'], use_column_width=True)
                        
                        # Adding extra space for Shirt (index 1) and T-Shirt (index 2)
                        if img_index == 1 or img_index == 2:
                            st.write("")  # Empty line for spacing

                        if st.button(f"Recommend {img_data['item']}"):
                            input_img_features = extract_features_from_images(img_data['path'], model)
                            distance, indices = neighbors.kneighbors([input_img_features])
                            recommended_images = [filenames[indices[0][j+1]] for j in range(5)]  # Skip the first one because it's the same image
        
        # Upload image feature
        upload_file = st.file_uploader("Upload Image")
        if upload_file is not None:
            upload_path = os.path.join('C:/Users/Lenovo/OneDrive/Desktop/mini proj/upload', upload_file.name)
            with open(upload_path, 'wb') as f:
                f.write(upload_file.getbuffer())
            st.subheader('Uploaded Image')
            st.image(upload_file)

            # Extract features from the uploaded image and find recommendations
            input_img_features = extract_features_from_images(upload_path, model)
            distance, indices = neighbors.kneighbors([input_img_features])
            recommended_images = [filenames[indices[0][j+1]] for j in range(5)]  # Skip the first one because it's the same image
        
        # Display recommended images at the bottom of the page
        if recommended_images:
            st.subheader('Recommended Products')
            rec_cols = st.columns(5)
            for j, rec_col in enumerate(rec_cols):
                with rec_col:
                    if j < len(recommended_images):
                        rec_img_path = recommended_images[j]
                        rec_img = Image.open(rec_img_path)
                        rec_description_data = descriptions.get(
                            os.path.basename(rec_img_path), 
                            {"productDisplayName": "No description available", "Price": "N/A", "Rating": "N/A", "usage": "N/A", "subCategory": "N/A"}
                        )
                        
                        st.image(rec_img, caption=rec_description_data['productDisplayName'], use_column_width=True)
                        st.write(f"**Price:** {rec_description_data['Price']}")  # Display the price for recommended products
                        st.write(f"**Rating:** {rec_description_data['Rating']}")  # Display the rating for recommended products
                        st.write(f"**Usage:** {rec_description_data['usage']}")  # Display the usage for recommended products
                        st.write(f"**SubCategory:** {rec_description_data['subCategory']}")  # Display the subCategory for recommended products
if __name__ == "__main__":
    main()