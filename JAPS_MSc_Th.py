import os

import tensorflow as tf
from tensorflow import keras as keras

from PIL import Image
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #Disable oneDNN custom operations used for otimization 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #Turn off messages about TensorFlow otimization operations


#Desktop
""" 
img_data_dir = "C:/Users/joaoa/Documents/[EU]Faculdade/Tese/SAR_Imaging_NNs_Thesis/GBSAR datset/RealSAR-IMG/"
raw_data_dir = "C:/Users/joaoa/Documents/[EU]Faculdade/Tese/SAR_Imaging_NNs_Thesis/GBSAR datset/RealSAR-RAW/"
"""
#Laptop
img_data_dir = "C:/Users/joaoa/Desktop/Tese/SAR_Imaging_NNs_Thesis/GBSAR datset/RealSAR-IMG/"
raw_data_dir = "C:/Users/joaoa/Desktop/Tese/SAR_Imaging_NNs_Thesis/GBSAR datset/RealSAR-RAW/"



deep_conv_encoder = keras.Sequential()

deep_conv_encoder.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = keras.activations.leaky_relu))
""" 
deep_conv_encoder.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

deep_conv_encoder.add(keras.layers.Conv2D(filters = 128, kernel_size = (3,3), activation = keras.activations.leaky_relu))

deep_conv_encoder.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

deep_conv_encoder.add(keras.layers.Conv2D(filters = 256, kernel_size = (3,3), activation = keras.activations .leaky_relu))

deep_conv_encoder.add(keras.layers.Conv2D(filters = 256, kernel_size = (3,3), activation = keras.activations.leaky_relu))

deep_conv_encoder.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

deep_conv_encoder.add(keras.layers.Conv2D(filters = 512, kernel_size = (3,3), activation = keras.activations.leaky_relu))

deep_conv_encoder.add(keras.layers.Conv2D(filters = 512, kernel_size = (3,3), activation = keras.activations.leaky_relu))

deep_conv_encoder.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

deep_conv_encoder.add(keras.layers.Conv2D(filters = 512, kernel_size = (3,3), activation = keras.activations.leaky_relu))

deep_conv_encoder.add(keras.layers.Conv2D(filters = 512, kernel_size = (3,3), activation = keras.activations.leaky_relu))

deep_conv_encoder.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

deep_conv_encoder.add(keras.layers.SpatialDropout2D(0.5)) #not sure if 0.5 is the correct dropout rate

deep_conv_encoder.add(keras.layers.Dense(2014, activation = keras.activations.leaky_relu))

deep_conv_encoder.add(keras.layers.Dense(2014, activation = keras.activations.leaky_relu))
 """

for i, layer in enumerate(deep_conv_encoder.layers):
    print(f'Layer {i}: {layer.name}, {layer.__class__.__name__}')

# Compile the model
deep_conv_encoder.compile(optimizer = 'adam', loss = 'mean_absolute_error')

# Print the model summary
deep_conv_encoder.summary()

def get_files_list (dir_path):
    
    files_names_list = os.listdir(dir_path)
    files_names_list.sort()

    return files_names_list


def img_to_numpy(image_path):
    
    # Open the image file
    img = Image.open(image_path)
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    return img_array


def txt_to_numpy(txt_file_path):
    
    # Open the image file
    txt_file = np.loadtxt(txt_file_path) 
    
    # Convert the image to a NumPy array
    txt_array = np.array(txt_file).reshape(len(txt_file), -1, 1)
    
    return txt_array


def import_img_files (dir_path, files_names):

    img_files = []
    for i in files_names:
        img_files.append(img_to_numpy(dir_path + i))

    img_files = np.asarray (img_files)

    return img_files

def import_txt_files (dir_path, files_names):

    txt_files = []
    for i in files_names:
        txt_files.append(txt_to_numpy(dir_path + i))

    txt_files = np.asarray (txt_files)

    return txt_files


img_files_names = get_files_list(img_data_dir)
raw_files_names = get_files_list(raw_data_dir)

for i in range(len(img_files_names)):

    if img_files_names[i][:-3] != raw_files_names[i][:-3]: 
        print (f"ERROR: File number {i} incoherent name\n\tIMG file name: {img_files_names[i]}\n\tRAW file name: {raw_files_names[i]}\nVerify if files correspond befores proceeding.")

img_files = import_img_files(img_data_dir, img_files_names)
raw_files = import_txt_files(raw_data_dir, raw_files_names)

print (img_files.shape)
print (raw_files.shape)

# Train the model
deep_conv_encoder.fit(raw_files, img_files, epochs = 80, batch_size = 32, validation_split = 0.2)



'''
image = tf.io.read_file('image.jpg')
image = tf.image.decode_image(image)

# Convert to NumPy array
image_array = image.numpy()

print(image_array.shape)



x = np.random.rand(1, height, width, channels)
y = keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = keras.activations.leaky_relu, input_shape=(height, width, channels))(x)

print(y.shape) # (batch_size, new_height, new_width, filters)

a = np.array ([
                [0, 9, 2, 3],
                [27, 28, 13, 12],
                [12, 9, 26, 21],
                [3, 26, 43, 18]
              ])

a = np.reshape(a, [1, 4, 4, 1])


z = keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(a)

print("============================ OUTPUT ============================")
print (z.numpy())
print (z.shape)
'''