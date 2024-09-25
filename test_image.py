from PIL import Image
import numpy as np

def png_to_numpy(image_path):
    # Open the image file
    img = Image.open(image_path)
    
    img.show()
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    return img_array

x_train = png_to_numpy("C:/Users/joaoa/Documents/[EU]Faculdade/Tese/SAR_Imaging_NNs_Thesis/GBSAR datset/RealSAR-IMG/0_Aluminium.png")

print(f"Image shape: {x_train.shape}")
