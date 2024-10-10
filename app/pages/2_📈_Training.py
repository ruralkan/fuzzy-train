import streamlit as st
import os
from pathlib import Path
import pandas as pd
import datetime
import requests
import pendulum
files = os.listdir('../models')
files.append("Add new model...")
st.info(os.listdir('../datasets'))
def uploader(selection, save_folder,uploaded_files):
    if uploaded_files:
        
        for file in uploaded_files:
            file.seek(0)
            # Save uploaded file to 'F:/tmp' folder.
            #save_folder = '../models'
            save_path = Path(save_folder, file.name)
            with open(save_path, mode='wb') as w:
                st.info(file.getbuffer())
                w.write(file.getbuffer())

            if save_path.exists():
                st.success(f'File {file.name} is successfully saved!')

    

st.markdown("# Trainining resource")
st.markdown("## At this time we are only use elasticnet_model.py to train the model, but you can manually change it")
st.markdown('''
        <p>You can use 3 different models to train that you can find in folder /include/models:<p>
            -  ElasticNet_ model.py
            -   randomforest_model.py
            -   SVM_model.py
        <p>If you want to use a different model you should change line 14 within the file /dags/mlflow_integration.py<p>
        <p>Ex. if you want to use randomforest model:<p>
        <p>import include.models.ElasticNet_model as model_class ---> import include.models.randomforest_model as model_class<p>
        <p>You can create new .py files to train more models<p>
                   ''')

## Create 
#st.markdown('Select a python file to train a model')

#selection = st.selectbox("Select option", options=files)
#selection_name = selection.split('.')[0]
## Create text input for user entry
#if selection == "Add new model...": 
#    otherOption = st.text_input("Enter your other option...")

#    uploaded_files = st.file_uploader("Choose a model", type=['py',], accept_multiple_files=True)
    
#    uploader(selection, '../models',uploaded_files)
#    # Just to show the selected option
#if selection != "Another option...":
#    st.info(f":white_check_mark: The selected option is {selection} ")

#else: 
#    st.info(f":white_check_mark: The written option is {otherOption} ")
     

#st.divider()
#st.markdown("Select a dataset to train the model")
#csv_files = os.listdir('../datasets')
#csv_files.append("Add new csv file...")
#csv_selection = st.selectbox("Select option", options=csv_files)

## Create text input for user entry
#if csv_selection == "Add new csv file...": 
#    otherOption = st.text_input("Enter your other option...")

#    uploaded_files = st.file_uploader("Choose a dataset", type=['csv',], accept_multiple_files=True)
    
#    uploader(csv_selection, '../datasets',uploaded_files)

#if csv_selection != "Another option...":
#    st.info(f":white_check_mark: The selected option is {csv_selection} ")

#else: 
#    st.info(f":white_check_mark: The written option is {otherOption} ")


if st.button("Train"):
    import requests

    from requests.auth import HTTPBasicAuth
    url='http://host.docker.internal:8080/api/v1/dags/wine_feature_eng/dagRuns'
    utc_time = pendulum.now('UTC')
    data = {
        "conf": {},
        "dag_run_id": 'new_run_at_' + utc_time.strftime('%Y-%m-%dT%H:%M:%S'),
        "logical_date": utc_time.strftime('%Y-%m-%dT%H:%M:%S')+'Z',
        "note": "Hello"
        }
    #st.info(data)
    response = requests.post(url, json=data, auth=("admin", "admin"))
    st.info(response)
    