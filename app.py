import streamlit as st
import tensorflow as tf
import numpy as np
from keras.models import load_model
import cv2



with open ("style.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)
    
#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("plant_disease_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(256,256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element
  
#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition","Treatment"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE DETECTION SYSTEM")
    image_path = "https://media.istockphoto.com/id/658291850/photo/young-plant-growing-in-sunlight.jpg?s=612x612&w=0&k=20&c=ZH9gmP8dLhwNaWuVbtBz99Fkybg_B0Uanw88QXSdMnY="
    st.image(image_path,use_column_width=True)
    st.markdown("""
   Welcome to the Plant Disease Setection System! 🌿🔍
    
   Plant Disease Detection is a state-of-the-art machine learning project that classifies and identifies plant diseases using convolutional neural networks (CNN) and deep learning techniques. The main goal is to give farmers and agricultural experts a useful tool for quickly identifying plant health, allowing for quick intervention, and reducing the risk of crop loss.
    
   Our goal is to effectively assist in the identification of plant diseases. Our technology can identify any indications of disease by analysing a plant image that you upload. Let's collaborate to protect our crops and ensure a more nutritious produce.


    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
    - **Early Detection.** Plant infections can be identified quickly, allowing for early action that can stop the disease's progress and reduce crop damage.
    - **Resource Efficiency.** Accurately identifying sick plants can assist farmers in making the most use of resources like pesticides, fertilisers, and water.
    - **Cost Savings.** By lowering the need for excessive pesticide usage and averting crop losses, early detection and focused treatment of plant diseases save farmers money.


    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, dataset, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    image_path = "https://storage.googleapis.com/kaggle-datasets-images/277323/573309/8a64442606a7e525f1df4819fe0126e5/dataset-cover.png?t=2019-07-27-16-20-38"
    st.image(image_path,use_column_width=True)
    st.markdown("""
              #### About Project 
              A comprehensive study that uses CNN and Deep Learning to identify and categorise diseases of plants will help specialists and farmers stop outbreaks and preserve agricultural productivity.

               ### Dataset
               -**Context.** PlantVillage dataset containing 38 classes of plants with around 54,303 images
                
               -**Content.**
                Plant Village dataset is a public dataset of 54,305 images of diseased and healthy plant leaves collected under controlled conditions ( PlantVillage Dataset). The images cover 14 species of crops, including: apple, blueberry, cherry, grape, orange, peach, pepper, potato, raspberry, soy, squash, strawberry and tomato. It contains images of 17 basic diseases, 4 bacterial diseases, 2 diseases caused by mold (oomycete), 2 viral diseases and 1 disease caused by a mite. 12 crop species also have healthy leaf images that are not visibly affected by disease.
                
                -**Model Training 🧠.**
                The model was trained using the Plant_Disease_Detection.ipynb notebook. It employs a Convolutional Neural Network architecture to classify plant images into different disease categories. The trained model weights are saved in plant_disease_model.h5.

                """)
    
    #Treatment
elif(app_mode=="Treatment"):
    st.header("Treatment")
    image_path = "https://media.licdn.com/dms/image/D4D12AQHRHdbN-40f7g/article-cover_image-shrink_720_1280/0/1675395957571?e=2147483647&v=beta&t=rRmIXQd5dwq1LBwnfoDu7BMGmuQIyJanVgwIGpqh1Qo"
    st.image(image_path,use_column_width=True)
    st.markdown("""
                ### Potato Barly Blight
                  **Meaning:**
                   Phytophthora infestans is a fungus-like organism that causes potato blight, a deadly disease that can also harm other plants like tomatoes. It causes the tubers to rot and quickly lose their foliage, which results in large yield losses.

                  **Lifecycle:**
                   Spores of the pathogen propagate by air and water during the life cycle of potato blight. Its development is aided by warm, humid weather. Usually, the illness begins as dark spots on leaves that spread fast to kill the foliage. Dark, decaying patches appear on infected tubers.

                  **Solutions:**
                     Planting resistant potato cultivars, using fungicides in advance, rotating crops to minimise pathogen accumulation, and maintaining adequate soil drainage to prevent waterlogging—which can encourage the spread of disease—are some methods for controlling potato blight.

                     
                ### Tomato-Bacterial_spot
                  **Meaning:**
                     Small, dark spots on the leaves, stems, and crops of tomato plants are the visible sign of tomato bacterial spot. With time, these spots may take on a raised, scabby texture in addition to their characteristic water-soaked appearance. Tomato crops that are severely damaged by the disease may have lower yields and worse quality crops.
                     
                  **Lifecycle:**
                     Plant waste, diseased plant material, and contaminated seeds are the main ways that tomato bacterial spot spreads. It can also spread through mechanical contact, wind, and rain. Through wounds or stomata, the bacterium enters the plant naturally. There, it grows and produces the distinctive spots. The disease is more likely to develop and spread in warm, humid climates, which makes it especially dangerous in those areas.
                     
                   **Solutions:**
                     If it's possible, use drip irrigation  rather than overhead irrigation. Give plants a morning watering so that the water will evaporate rapidly. Steer clear of handling moist plants. Pesticide: Copper products, or copper plus mancozeb, are approved and work well for controlling tomato bacterial spots in homes.
                     
                ### Corn-Common_rust
                   **Meaning:**
                     A common plant disease that affects corn harvests all around the world is called common rust of corn, and it is brought on by the fungus Puccinia sorghi. It appears on the leaves as tiny, elevated pustules that are rounded to elongated and contain large numbers of rust-colored spores. Reduced photosynthesis, slowed growth, and decreased production can result from severe infections.
                     
                   **Lifecycle:**
                     The first stage of the life cycle of corn common rust is the germination of spores on maize leaves, which usually happens in warm, humid weather. The spores enter the plant through the leaf surface, infecting the tissue and producing the recognisable pustules. More spores are produced by these pustules, and when these spores are dispersed to nearby plants by wind and rain, fresh infections are started. The fungus completes its life cycle by overwintering on different hosts, like common grasses.
                     
                   **Solutions:**
                     For high-value crops like popcorn, sweet corn and seed corn, foliar spraying of fungicides is the usual method of controlling sorghi. Disease-resistant hybrids are also used on maize.


    """)
    
#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
        
#Treatment dictionary (replace with actual treatment information)
    treatments = {
    'Tomato-Bacterial_spot': 'Use copper-based bactericides or neem oil spray.',
    'Potato-Barly blight': 'Remove infected parts and apply copper-based fungicides.',
    'Corn-Common_rust': 'Apply fungicides containing copper or sulfur.',
    'Healthy': 'No treatment necessary, focus on preventive measures.',
}
        
#Predict button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        #Reading Labels
        class_name = ['Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
        st.markdown(f'## Treatment Recommendation:')
        st.info(treatments.get(class_name[result_index],'Treatment information unavailable.'))

    
    
       
        
        
        