import streamlit as st
from io import BytesIO
from keras.models import load_model
import cv2
import numpy as np
import os
import pandas as pd
st.title("Fruits🍅 Classification")
df_confidence = pd.DataFrame(columns=['Fruit', 'Confidence'])
if 'stored_data' not in st.session_state:
    st.session_state.stored_data = []
def display_fruit_info(result, label_name, verify_result=None):
    # Display confidence for the selected fruit in a table
    global df_confidence
   
    st.sidebar.title('Display fruits info')
    label_confidence = [
        {'Fruit': label_name[i], 'Confidence': f"{round(result[i] * 100, 2):.2f}%".rstrip('0').rstrip('.')}
    for i in range(len(result))
    if result[i] > 0.3
    ]
    label_confidence.sort(key=lambda x: x['Confidence'])
    if verify_result is not None and verify_result < len(label_confidence):
        st.sidebar.success(f'**Predicted Label:** {label_confidence[verify_result]["Fruit"]}')
        st.sidebar.info(f'**Confidence:** {label_confidence[verify_result]["Confidence"]}')
    # Displaying the label and confidence information in a table
    df = pd.DataFrame(label_confidence)
    st.sidebar.table(df)
    # Always store data to the session state
    st.session_state.stored_data.append(label_confidence.copy())
    update_stored_data_table()

def update_stored_data_table():
    if st.session_state.stored_data:
        st.sidebar.write("Stored Data:")
        data = []
        for entry in st.session_state.stored_data:
            for label_info in entry:
                # Format confidence values
                label_info['Confidence'] = f"{float(label_info['Confidence'].rstrip('%')):.2f}%"
                data.append(label_info)
        
        stored_df = pd.DataFrame(data)
        st.sidebar.write(stored_df)
Enter_kcal = st.number_input("Enter Kcal you need:", min_value=0, value=None)
st.write('Your chosen kcal is', Enter_kcal, 'kcal')
img_array = []
path_img = r"D:\datasheet\new fruits data"
for file in os.listdir(path_img):
    img = os.path.join(path_img, file)
    print(img)
    img_read = cv2.imread(img)
    img_read = cv2.resize(img_read, (500, 500))
    img_array.append(img_read)
kcal = [71, 52, 57, 57, 57, 63, 52, 52, 52, 72,
       57,48, 160, 198, 89, 89, 34, 34, 31, 47,
        17, 17, 90, 67, 67, 67, 43, 33, 68, 65,
        71, 43, 20, 30, 67, 66, 97, 39, 62, 43,
        35, 57, 46, 57, 50, 50, 60, 46, 53, 77,
        32, 47
   ]
label_data = np.load("label_data.npy")
label_name = np.load("label_name.npy")
data = np.load("data.npy")
model = load_model("model_CNN_1.h5")
path_img = st.file_uploader("Choose an Image", type=["jpg", "png"])
if path_img is not None:
    image_bytes = path_img.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, (50, 50))
    normalized_image = resized_image.astype('float32') / 255
    reshaped_image = normalized_image.reshape((1, 50, 50, 3))
    result = model.predict(reshaped_image)
    result = np.round(result, 2)
    result = result[0]
    print(result)
    display_fruit_info(result, label_name)
    for verify_result in range(len(result)):
     if result[verify_result] > 0.6:
        if result[verify_result] == 1.0:
            img_array = img_array[verify_result]
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            st.image(img_array, width=400)
            st.success(f'**Predicted Label:** {label_name[verify_result]}',)
            st.warning(f'**Kcal:** {kcal[verify_result]}')
            st.info(f'**Confidence:** {np.round(result[verify_result] * 100, decimals=2)} %')
           #  bản đồ tính confidence %
            confidence_data = {
                    'Fruit': label_name,
                    'Confidence': np.round(result * 100, decimals=2)
                }
            df_confidence = pd.DataFrame(confidence_data)
            Kcal_you_need = round(Enter_kcal / kcal[verify_result])
            kg = np.round((Kcal_you_need*0.1), decimals=2)
            g = np.round((kg * 1000), decimals=2)
            st.info(f'**Estimated Quantity to Buy:** {Kcal_you_need} {label_name[verify_result]}')
            st.info(f'**Estimated Weight:** {kg} kg')
            st.info(f'**Estimated Weight:** {g} g')
        else:
            img_array = img_array[verify_result]
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            st.image(img_array, width=400)
            st.success(f'**Predicted Label:** {label_name[verify_result]}')
            st.warning(f'**Kcal:** {kcal[verify_result]}')
            st.info(f'**Confidence:** {np.round(result[verify_result] * 100, decimals=2)} %')
       #  bản đồ tính confidence %
            confidence_data = {
                    'Fruit': label_name,
                    'Confidence': np.round(result * 100, decimals=2)
                }
            df_confidence = pd.DataFrame(confidence_data)
            st.line_chart(df_confidence.set_index('Fruit')['Confidence'])
            Kcal_you_need = round(Enter_kcal / kcal[verify_result])
            kg = np.round( Kcal_you_need*0.1, decimals=2)
            g = np.round((kg * 1000), decimals=2)
            st.info(f'**Estimated Quantity to Buy:** {Kcal_you_need} {label_name[verify_result]}')         
            st.info(f'**Estimated Weight:** {kg} kg')
            st.info(f'**Estimated Weight:** {g} g')
else:
       st.warning('Please upload an image.')
cv2.waitKey(0)




