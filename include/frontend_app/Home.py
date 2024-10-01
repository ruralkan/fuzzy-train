import streamlit as st
import boto3

#s3 = boto3.resource('s3')
#BUCKET = "test"


#s3.Bucket(BUCKET).upload_file("your/local/file", "dump/file")

import streamlit as st
from PIL import Image


st.title('The Prototype')
st.header('Training, Evaluation and Monitorin plataform')
st.markdown('Building a Prototype for the MLOps Certifcation.')

st.divider()

col1, col2 = st.columns(2)

with col1:
   st.subheader("Training")
   forecasting_image = Image.open('./assets/training.png')
   st.image(forecasting_image)
   st.caption('Train a Regression Model ElasticNet with L1_ratio and Alpha')
   
with col2:
   st.subheader('Monitoring Dashboard')
   forecasting_image = Image.open('./assets/stats.png')
   st.image(forecasting_image)
   st.caption('Visualization of several metrics of models')

st.divider()