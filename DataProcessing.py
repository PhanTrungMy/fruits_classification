import os
import cv2
import numpy as np
from tqdm import tqdm

path = r"D:\DATA\new_data"
label_name = []
label_data = []
data = []
for folder in tqdm(os.listdir(path), desc="Processing"):
    file = os.path.join(path, folder)
    label_name.append(folder)
    for i in os.listdir(file):
        img = os.path.join(file, i)
        try:
            image = cv2.imread(img)
            if image is not None:  # Check if the image is loaded properly
                image = cv2.resize(image, (50,50))
                data.append(image)
                label_data.append(folder)
            else:
                print(f"Skipping {img} as it couldn't be read.")
        except Exception as e:
            print(f"Error processing {img}: {str(e)}")

np.save("label_name_2.npy", label_name)
np.save("data_2.npy", data)
np.save("label_data_2.npy", label_data)
x= np.load("label_name_2.npy")
y= np.load("data_2.npy")
z= np.load("label_data_2.npy")
print(x.shape)
print(y.shape)
print(z.shape)














