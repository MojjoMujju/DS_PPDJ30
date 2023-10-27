import streamlit as st
import numpy as np
from lightgbm import LGBMClassifier

#load ML package
import joblib
import os

attribute_info = """
                 Features:
                1) id: unique identifier
                2) gender: "Male", "Female" or "Other"
                3) hypertension: 
                    - 0 = if the patient doesn't have hypertension, 
                    - 1 = if the patient has hypertension
                4) ever_married: "No" or "Yes"
                5) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
                6) Residence_type: "Rural" or "Urban"
                7) avg_glucose_level: average glucose level in blood
                8) bmi: body mass index
                    - Below 18.5 -> Underweight
                    - 18.5 - 24.9 -> Normal or Healthy Weight
                    - 25.0 - 29.9 -> Overweight
                    - 30.0 and Above -> Obese
                https://www.cdc.gov/healthyweight/assessing/index.html
                
                9) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
                10) age: age of the patient : from 17 Up
                 """

gen = {'Male':0, 'Female':1}
hyper = {'No': 0, 'Yes':1}
married = {'Ever': 0, 'Never': 1}
work = {'Self-employed':0, 'Govt_job':1, 'Private':2}
Residence = {'Rural':0,'Urban':1}
glucose_level = {'Normal':0, 'Pra-Diabetes':1, 'Diabetes':2}
bmi_ = {'Healthy':0, 'Overweight':1, 'Obese':2}
smoke = {'never smoked':0, 'Unknown':1, 'formerly smoked':2, 'smokes':3}
        
def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

@st.cache(allow_output_mutation=True)
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),'rb'))
    return loaded_model


def run_ml_app():
    st.subheader("ML section")

    with st.expander("Attribute Info"):
        st.markdown(attribute_info)

    st.subheader("Input Your Data")
    gender = st.radio('Gender', ["Male", "Female"])
    hypertension = st.radio('Hypertension', ["No", "Yes"])
    ever_married = st.radio('Ever Married', ["Never", "Ever"])
    work_type = st.selectbox('Work Type', ["Self-employed", "Govt_job", "Private"])
    Residence_type = st.radio('Residence Type', ["Rural", "Urban"])
    avg_glucose_level = st.selectbox('Average Glucose Level', ["Normal", "Pra-Diabetes", "Diabetes"])
    bmi = st.selectbox('BMI', ["Healthy", "Overweight", "Obese"])
    smoking_status = st.selectbox('Smoking Status', ["never smoked", "Unknown", "formerly smoked", "smokes"])
    age = st.number_input('Age', 18,100)

    with st.expander("Your Selected Options"):
        result = {
            'Gender':gender,
            'Hypertension':hypertension,
            'Ever Married':ever_married,
            'Work Type':work_type,
            'Residence Type':Residence_type,
            'Average Glucose Level':avg_glucose_level,
            'BMI':bmi,
            'Smoking Status':smoking_status,
            'Age':age,
        }
    
    st.write(result)

    encoded_result = []
    for i in result.values():
        if type(i) == int:
            encoded_result.append(i)
        elif i in ["Male", "Female"]:
            res = get_value(i, gen)
            encoded_result.append(res)
        elif i in ["No", "Yes"]:
            res = get_value(i, hyper)
            encoded_result.append(res)
        elif i in ["Never", "Ever"]:
            res = get_value(i, married)
            encoded_result.append(res)
        elif i in ["Self-employed", "Govt_job", "Private"]:
            res = get_value(i, work)
            encoded_result.append(res)
        elif i in ["Rural", "Urban"]:
            res = get_value(i, Residence)
            encoded_result.append(res)
        elif i in ["Normal", "Pra-Diabetes", "Diabetes"]:
            res = get_value(i, glucose_level)
            encoded_result.append(res)
        elif i in ["Healthy", "Overweight", "Obese"]:
            res = get_value(i, bmi_)
            encoded_result.append(res)
        elif i in ["never smoked", "Unknown", "formerly smoked", "smokes"]:
            res = get_value(i, smoke)
            encoded_result.append(res)

    st.write(encoded_result)

    ## prediction section
    st.subheader('Prediction Result')
    single_sample = np.array(encoded_result).reshape(1,-1)
    
    #st.write(single_sample)

    model = load_model("model_LGBM.pkl")

    prediction = model.predict(single_sample)
    pred_proba = model.predict_proba(single_sample)
    #st.write(prediction)
    #st.write(pred_proba)

    pred_probability_score = {'Heart Diseases':round(pred_proba[0][1]*100,4),
                                'Healty':round(pred_proba[0][0]*100,4)}

    if prediction == 1:
        st.success("Terdapat indikasi adanya Heart Disease")
        
        st.write(pred_probability_score)
    else:
        st.warning("Tidak ada indikasi adanya Heart Disease")
        st.write(pred_probability_score)