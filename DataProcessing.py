import os
import cv2
import numpy as np
from tqdm import tqdm

path = r"/content/drive/MyDrive/DATA-BASIC/Data"
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
                image = cv2.resize(image, (10,10))
                data.append(image)
                label_data.append(folder)
            else:
                print(f"Skipping {img} as it couldn't be read.")
        except Exception as e:
            print(f"Error processing {img}: {str(e)}")

np.save("label_name1.npy", label_name)
np.save("label_data1.npy", label_data)
np.save("data1.npy", data)
        















