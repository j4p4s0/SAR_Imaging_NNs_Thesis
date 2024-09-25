import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #Disable oneDNN custom operations used for otimization 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #Turn off messages about TensorFlow otimization operations

import tensorflow as tf
from tensorflow import keras as keras

from PIL import Image
import numpy as np

# Function to load a PNG image and convert it to a NumPy array


deep_conv_encoder = keras.Sequential()

deep_conv_encoder.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = keras.activations.leaky_relu))

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


for i, layer in enumerate(deep_conv_encoder.layers):
    print(f'Layer {i}: {layer.name}, {layer.__class__.__name__}')

# Compile the model
deep_conv_encoder.compile(optimizer = 'adam', loss = 'mean_absolute_error')

# Print the model summary
deep_conv_encoder.summary()

def png_to_numpy(image_path):
    # Open the image file
    img = Image.open(image_path)
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    return img_array

x_train = png_to_numpy("GBSAR dataset/RealSAR-IMG/0_Aluminium.png")
x_train = np.append(x_train, png_to_numpy("GBSAR dataset/RealSAR-IMG/1_Aluminium.png"))


# Train the model
#deep_conv_encoder.fit(x_train, y_train, epochs = 80, batch_size = 32, validation_split = 0.2)



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