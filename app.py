# main Python app
import streamlit as st
import streamlit.components.v1 as stc

#import our app
from ml_app import run_ml_app

def main():
    st.title("Main App")

    menu = ["Home","Machine Learning"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown("""
            ## Risk Factors Prediction of Heart  Disease

            This dataset is used to predict whether a patient is likely to get heart disease based on the input parameters like gender, age, hypertension, ever married, work type, residence type. average glucose level, bmi,  and smoking status. Each row in the data provides relavant information about the patient.

            ## Independent Column
            1) id: unique identifier
            2) gender: "Male", "Female" or "Other"
            3) age: age of the patient
            4) hypertension: 0=if the patient doesn't have hypertension, 1=if the patient has hypertension
            5) ever_married: "No" or "Yes"
            6) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
            7) Residence_type: "Rural" or "Urban"
            8) avg_glucose_level: average glucose level in blood
            9) bmi: body mass index
                - Below 18.5 -> Underweight
                - 18.5 - 24.9 -> Normal or Healthy Weight
                - 25.0 - 29.9 -> Overweight
                - 30.0 and Above -> Obese
            https://www.cdc.gov/healthyweight/assessing/index.html

            10) smoking_status: "formerly smoked",
                "never smoked", "smokes" or "Unknown"*
                    
            ## Dependent Column
            heart_disease: 0= if the patient doesn't have any heart diseases, 1= if the patient has a heart disease

            ## ML Model Random Forest Classification
            """)
        
    elif choice == "Machine Learning":
        run_ml_app()
    

if __name__ == '__main__':
    main()