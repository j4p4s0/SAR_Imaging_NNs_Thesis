from PIL import Image
import numpy as np

import os

img_data_dir = "C:/Users/joaoa/Documents/[EU]Faculdade/Tese/SAR_Imaging_NNs_Thesis/GBSAR datset/RealSAR-IMG/"
raw_data_dir = "C:/Users/joaoa/Documents/[EU]Faculdade/Tese/SAR_Imaging_NNs_Thesis/GBSAR datset/RealSAR-RAW/"


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

    img_files = np.asarray(img_files)

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

print(f"Images Array Shape: {img_files.shape}")
print(f"Raw Array Shape: {raw_files.shape}")
