import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
#Extract Filenames from Folder
filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))
len(filenames)
#Importing ResNet50 Model and Cofiguration
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tf.keras.models.Sequential([model,
                                   GlobalMaxPool2D()
                                   ])
model.summary()
#Extracting Fetaures from Image
img = image.load_img('C:/Users/Lenovo/OneDrive/Desktop/mini proj/images/1163.jpg', target_size=(224,224))
img_array = image.img_to_array(img)
img_expand_dim = np.expand_dims(img_array, axis=0)
img_preprocess = preprocess_input(img_expand_dim)
result = model.predict(img_preprocess).flatten()
norm_result = result/norm(result)
norm_result
def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result/norm(result)
    return norm_result
extract_features_from_images(filenames[0], model)
image_features = []
for file in filenames:
    image_features.append(extract_features_from_images(file, model))
image_features
Image_features = pkl.dump(image_features, open('embeddings.pkl','wb'))
filenames = pkl.dump(filenames, open('filenames.pkl','wb'))
#Loading Pickle Files
Image_features = pkl.load(open("C:/Users/Lenovo/OneDrive/Desktop/mini proj/embeddings.pkl",'rb'))
filenames = pkl.load(open('filenames.pkl','rb'))
np.array(Image_features).shape
#Finidng Simialar Images
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)
input_image = extract_features_from_images("C:/Users/Lenovo/OneDrive/Desktop/mini proj/images/1526.jpg",model)
print(input_image)
distance,indices = neighbors.kneighbors([input_image])
indices[0]
from IPython.display import Image
Image('16871.jpg')
Image(filenames[indices[0][1]])
Image(filenames[indices[0][2]])
Image(filenames[indices[0][3]])
Image(filenames[indices[0][4]])
Image(filenames[indices[0][5]])
