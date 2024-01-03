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
path_img = r"D:\DATA\text"
for file in os.listdir(path_img):
    img = os.path.join(path_img, file)
    print(img)
    img_read = cv2.imread(img)
    img_read = cv2.resize(img_read, (500, 500))
    img_array.append(img_read)

kcal = [52, 160, 89, 35, 53, 29, 38, 75, 20, 43 ]
water = [86, 73, 75, 90, 80, 89, 75, 74, 70, 75   ]
carbs = [13.8, 8.5, 22.84, 14.4, 12, 9.3, 9.62,16, 3.92,11.1]
fat = [0.2, 14, 0.03, 0.2, 0.3, 0.3, 0.4, 0.03, 2, 0.3, 0.2 ]
fiber = [2.4, 2.6, 1.6, 0.9, 1.5,2.8, 1, 1.4, 1.2, 0.3 ]
shopping_link = {
    'apple': 'https://www.bachhoaxanh.com/trai-cay-tuoi-ngon/tao-ambrosia-my-09-11kg?preFreshType=normal',
    'avocado': 'https://www.bachhoaxanh.com/trai-cay-tuoi-ngon/bo-tui-1kg-3-trai?preFreshType=normal',
    'banana': 'https://www.bachhoaxanh.com/trai-cay-tuoi-ngon/chuoi-gia-giong-nam-my-3kg?preFreshType=normal',
    'cantaloupe': 'https://www.bachhoaxanh.com/trai-cay-tuoi-ngon/dua-luoi-tron-ruot-cam-tui-1-trai-tu-11kg-tro-len?preFreshType=normal',
    'lemon': 'https://www.bachhoaxanh.com/cu/chanh-khong-hat-tui-250g-2-4-trai?preFreshType=expired',
    'pomelo': 'https://www.bachhoaxanh.com/trai-cay-tuoi-ngon/buoi-nam-roi-tui-1-trai?preFreshType=normal',
    'rambutan': 'https://www.bachhoaxanh.com/trai-cay-tuoi-ngon/chom-chom-thai-lan-1kg?preFreshType=normal',
    'tomato': 'https://www.bachhoaxanh.com/cu/ca-chua-beef-500g?preFreshType=normal',
    'yali pear': 'https://www.bachhoaxanh.com/trai-cay-tuoi-ngon/le-hoang-kim-1kg?preFreshType=normal',  
}

label_data = np.load("label_data_new.npy")
label_name = np.load("label_name_new.npy")
data = np.load("data_new.npy")
model = load_model("model_new1.h5")

path_img = st.file_uploader("Choose an Image", type=["jpg", "png"])

if path_img is not None:
    image_bytes = path_img.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, (10, 10))
    normalized_image = resized_image.astype('float32') / 255
    reshaped_image = normalized_image.reshape((1, 10, 10, 3))
    result = model.predict(reshaped_image)
    result = np.round(result, 2)
    result = result[0]
    print(result)
    display_fruit_info(result, label_name)
    for verify_result in range(min(len(result), len(label_name))):
     if result[verify_result] > 0.3:
        if result[verify_result] == 1.0:
            
            img_array = img_array[verify_result]
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            st.image(image_bytes, width=400)
            # st.success(f'**Predicted Label:** {label_name[verify_result]}',)
            fruit_prediction = label_name[verify_result]
            st.success(f'**Predicted Label:** {fruit_prediction}')
            # hiển thị thông tin dữ liệu
            selected_data = {
                'Fruit': label_name[verify_result],
                'Calories(cal)': [kcal[verify_result]],
                'water(%)': [water[verify_result]],
                'Fat(g)': [fat[verify_result]],
                'Carbohydrate(g)': [carbs[verify_result]],
                'Fiber(g)': [fiber[verify_result]]
            }
            df_selected = pd.DataFrame(selected_data)
            st.table(df_selected)
        #  bản đồ tính confidence %
            confidence_data = {
                    'Fruit': label_name,
                    'Confidence': np.round(result * 100, decimals=2)
                }
            df_confidence = pd.DataFrame(confidence_data)
            st.line_chart(df_confidence.set_index('Fruit')['Confidence'])
            Kcal_you_need = round(Enter_kcal / kcal[verify_result])
            kg = np.round((Kcal_you_need*0.1), decimals=2)
            g = np.round((kg * 1000), decimals=2)
            st.info(f'**Estimated Quantity to Buy:** {Kcal_you_need} {label_name[verify_result]}')
            st.info(f'**Estimated Weight:** {kg} kg')
            st.info(f'**Estimated Weight:** {g} g')
            print("Predicted Label:", label_name[np.argmax(result)])
            if fruit_prediction in shopping_link:
                shop_link = shopping_link[fruit_prediction]
                st.markdown(
                    f'<a href="{shop_link}" target="_blank" style="text-decoration: none;">'
                    f'<button style="background-color: #4CAF50; border: none; color: white; padding: 10px 20px; text-align: center; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 5px;">{fruit_prediction} Shop</button>'
                    '</a>', 
                    unsafe_allow_html=True
                )
                st.markdown("""
    <style>
    body {
        background-color: #f0f0f0;
    }
    </style>
    """, unsafe_allow_html=True)
        else:
            img_array = img_array[verify_result]
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            st.image(image_bytes, width=400)
            # st.success(f'**Predicted Label:** {label_name[verify_result]}')
            fruit_prediction = label_name[verify_result]
            st.success(f'**Predicted Label:** {fruit_prediction}')
            selected_data = {
                'Fruit': label_name[verify_result],
                'Calories(cal)': [kcal[verify_result]],
                'water(%)': [water[verify_result]],
                'Fat(g)': [fat[verify_result]],
                'Carbohydrate(g)': [carbs[verify_result]],
                'Fiber(g)': [fiber[verify_result]]
            }
            df_selected = pd.DataFrame(selected_data)
            st.table(df_selected)
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
            print("Predicted Label:", label_name[np.argmax(result)])
        if fruit_prediction in shopping_link:
                shop_link = shopping_link[fruit_prediction]
                st.markdown(
                    f'<a href="{shop_link}" target="_blank" style="text-decoration: none;">'
                    f'<button style="background-color: #4CAF50; border: none; color: white; padding: 10px 20px; text-align: center; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 5px;">{fruit_prediction} Shop</button>'
                    '</a>', 
                    unsafe_allow_html=True
                )
else:
       st.warning('Please upload an image.')
cv2.waitKey(0)


