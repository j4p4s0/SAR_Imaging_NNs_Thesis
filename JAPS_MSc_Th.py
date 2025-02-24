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
#img_data_dir = "C:/Users/joaoa/Desktop/Tese/SAR_Imaging_NNs_Thesis/GBSAR datset/RealSAR-IMG/"
#raw_data_dir = "C:/Users/joaoa/Desktop/Tese/SAR_Imaging_NNs_Thesis/GBSAR datset/RealSAR-RAW/"


stride_x = 1
stride_y = 2

deep_conv_encoder = keras.Sequential()

deep_conv_encoder.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = keras.activations.leaky_relu))

deep_conv_encoder.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (stride_x, stride_y)))

deep_conv_encoder.add(keras.layers.Conv2D(filters = 128, kernel_size = (3,3), activation = keras.activations.leaky_relu))

deep_conv_encoder.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (stride_x, stride_y)))

deep_conv_encoder.add(keras.layers.Conv2D(filters = 256, kernel_size = (3,3), activation = keras.activations .leaky_relu))

deep_conv_encoder.add(keras.layers.Conv2D(filters = 256, kernel_size = (3,3), activation = keras.activations.leaky_relu))

deep_conv_encoder.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (stride_x, stride_y)))

deep_conv_encoder.add(keras.layers.Conv2D(filters = 512, kernel_size = (3,3), activation = keras.activations.leaky_relu))

deep_conv_encoder.add(keras.layers.Conv2D(filters = 512, kernel_size = (3,3), activation = keras.activations.leaky_relu))

deep_conv_encoder.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (stride_x, stride_y)))

deep_conv_encoder.add(keras.layers.Conv2D(filters = 512, kernel_size = (3,3), activation = keras.activations.leaky_relu))

deep_conv_encoder.add(keras.layers.Conv2D(filters = 512, kernel_size = (3,3), activation = keras.activations.leaky_relu))

deep_conv_encoder.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (stride_x, stride_y)))

deep_conv_encoder.add(keras.layers.SpatialDropout2D(0.5)) #not sure if 0.5 is the correct dropout rate

deep_conv_encoder.add(keras.layers.Dense(2014, activation = keras.activations.leaky_relu))

deep_conv_encoder.add(keras.layers.Dense(2014, activation = keras.activations.leaky_relu))


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

def normalize_raw_data (raw_data):
    
    # Normalizar os valores da matriz pelo valor máximo de magnitude
    # normalized_raw_data = raw_data / np.max(np.abs(raw_data))
    normalized_raw_data = raw_data

    # Separar a parte real e imaginária e armazenar em uma nova matriz n x n x 2
    real_part = np.real(normalized_raw_data)
    imag_part = np.imag(normalized_raw_data)

    # Criar a matriz n x n x 2
    new_raw_data = np.stack((real_part, imag_part), axis=-1)

    return new_raw_data

def import_raw_data (file_path):
    
    # Lê o arquivo e cria um array numpy a partir do conteúdo
    with open(file_path, 'r') as file:
        # Lê o conteúdo do arquivo e separa as linhas
        lines = file.readlines()

    # Converte as linhas em uma lista de listas
    data_i_notation = [line.strip().split(',') for line in lines]

    data_j_notation = [[c.replace('i', 'j') for c in row] for row in data_i_notation]

    return np.array(data_j_notation, dtype=complex)

def main ():

    folder_path = os.path.dirname(os.path.abspath(__file__))

    # Cria o caminho completo para o arquivo
    raw_file_path = os.path.join(folder_path, "raw_data.txt")
    img_file_path = os.path.join(folder_path, "ground-truth_s_Terrain.bmp")

    raw_data = import_raw_data(raw_file_path)

    img = img_to_numpy(img_file_path)

    new_raw_data = normalize_raw_data(raw_data)

    print (isinstance(raw_data, np.ndarray))
    print (isinstance(img, np.ndarray))

    print (raw_data.shape)
    print (new_raw_data.shape)

    '''  
    img_files_names = get_files_list(img_data_dir)
    raw_files_names = get_files_list(raw_data_dir)

    for i in range(len(img_files_names)):

        if img_files_names[i][:-3] != raw_files_names[i][:-3]: 
            print (f"ERROR: File number {i} incoherent name\n\tIMG file name: {img_files_names[i]}\n\tRAW file name: {raw_files_names[i]}\nVerify if files correspond befores proceeding.")

    img_files = import_img_files(img_data_dir, img_files_names)
    raw_files = import_txt_files(raw_data_dir, raw_files_names)

    '''
    
    '''
    num_samples = 3

    raw_data_list = []

    for i in range (num_samples):
        raw_data_list.append (raw_data)
    
    '''



    # Train the model
    deep_conv_encoder.fit(x = new_raw_data, y = img, verbose = 2, epochs = 80, batch_size = 32, validation_split = 0.2)

main()
