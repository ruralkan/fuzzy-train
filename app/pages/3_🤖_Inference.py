import streamlit as st
import requests
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import matplotlib.pyplot as plt


st.set_page_config(page_title="Inference", page_icon="🌍")

st.markdown("# Inferece engine")


client = mlflow.tracking.MlflowClient(tracking_uri='http://host.docker.internal:5000')
experiment_id = '5'
experiments = client.search_runs(experiment_ids=experiment_id, order_by=['metrics.mae']).to_list()

table=list()
for experiment in experiments:
    raw_dict = {'run_name':experiment.to_dictionary()['info']['run_name']}
    
    raw_dict.update(experiment.to_dictionary()['data']['params'])
    raw_dict.update(experiment.to_dictionary()['data']['metrics'])
    #raw_dict.update(experiment.to_dictionary()['data']['sklearn']['model_size_bytes'])
    table.append(raw_dict)

df = pd.DataFrame(table)
st.divider()
st.markdown("If you want do a inference of a sample introduce the values")# Learn, decide and get model from mlflow model registry
model_name = "wine_quality"
model_version = 1
model = mlflow.sklearn.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

import streamlit as st
formbtn = st.button("Predict one time")

if "formbtn_state" not in st.session_state:
    st.session_state.formbtn_state = False

if formbtn or st.session_state.formbtn_state:
    st.session_state.formbtn_state = True
    
    st.subheader("Features form")
    # name = st.text_input("Name")
    with st.form(key = 'Wine features'):
        st.write('Wine features')
    
        fixed_acidity  = st.number_input(label="fixed acidity ")
        volatile_acidity = st.number_input(label='volatile acidity')
        citric_acid = st.number_input(label='citric acid')
        residual_sugar = st.number_input(label='residual sugar')
        chlorides = st.number_input(label='chlorides')
        free_sulfur_dioxide = st.number_input(label='free sulfur dioxide')
        total_sulfur_dioxide = st.number_input(label='total sulfur dioxide')
        density = st.number_input(label='density')
        pH = st.number_input(label='pH')
        sulphates = st.number_input(label='sulphates')
        alcohol = st.number_input(label='alcohol')
        
    
        submit_form = st.form_submit_button(label="Predict", help="Click to predict the quality!")
    
        # Checking if all the fields are non empty
        if submit_form:
            X_new= pd.DataFrame([[fixed_acidity,volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, 
                                  total_sulfur_dioxide, density, pH, sulphates,  alcohol]])
            
            y_pred_new = model.predict(X_new)
            
            st.write(submit_form)
    
            if y_pred_new:
                # add_user_info(id, name, age, email, phone, gender)
                st.success(
                            f"Predicted value {y_pred_new}"
                        )
            else:
                st.warning("Please fill all the fields")

st.divider()
st.markdown("If you have a DataFrame from a batch please upload")

uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)

     
def extract(file_to_extract):
    if file_to_extract.name.split(".")[-1] == "csv": 
        extracted_data = pd.read_csv(file_to_extract)

    elif file_to_extract.name.split(".")[-1] == 'json':
         extracted_data = pd.read_json(file_to_extract, lines=True)

    elif file_to_extract.name.split(".")[-1] == 'xml':
         extracted_data = pd.read_xml(file_to_extract)
         
    return extracted_data

dataframes = []


if uploaded_files:
    for file in uploaded_files:
        file.seek(0)
        df = extract(file)
        dataframes.append(df)

    if len(dataframes) >= 1:
        merged_df = pd.concat(dataframes, ignore_index=True, join='outer')

    st.dataframe(merged_df)

    st.info('Charging model info')
    st.info("Predicting")
    X_new = merged_df.drop(["quality"], axis=1)
    y_pred_new = model.predict(X_new)
    st.info("This is the prediction")
    st.dataframe(y_pred_new)