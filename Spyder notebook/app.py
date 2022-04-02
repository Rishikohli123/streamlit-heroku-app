from pycaret.regression import load_model, predict_model

import streamlit as st

import pandas as pd
import numpy as np

model = load_model('Deployment_02042022')

def run():
    from PIL import Image
    rishi_image = Image.open('Rishi.jpg')
    st.title('Insurance Application')
    st.sidebar.image(rishi_image)
    st.sidebar.info('This app is created by Rishi as test')
    
    #Capture the Inputs
    Age =st.number_input('Age', min_value=1, max_value=100, value=25)
    Gender = st.selectbox('sex', ['male','female'])
    BMI =st.number_input('bmi', min_value=10, max_value=50, value=10)
    Children = st.selectbox('children', [0,1,2,3,4,5,6,7,8,9,10])
    if st.checkbox('Smoker'):
        smoker='yes'
    else:
        smoker='no'
        
    Region = st.selectbox('region', ['northwest','southwest','northeast','southeast'])
    
    Result = ""
    
    #Input dictionary
    Input_dict = {'age':Age, 'sex':Gender, 'bmi': BMI,'children': Children,'smoker': smoker,
                  'region': Region}
    input_df = pd.DataFrame([Input_dict])
    
    #Predict
    
    if st.button('Predict the results'):
        Result = predict_model(model, data = input_df)
        Result = '$' + str(Result['Label'][0])
        
    #Display the results
    st.success('The Insurance Amount is {}'.format(Result))
                    


if __name__=='__main__':
    run()