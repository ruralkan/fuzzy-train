import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to CF_Project! 👋")

st.sidebar.success("Select a option above.")

st.markdown(
    """
    This is a mockup to show the integretion between airflow, mlflow, streamlit 
    **👈 Select a option from the sidebar** 
    ### Monitoring
    See a complete dashboard about metrics by project like:

    ### Training
    Select a model and try to run a new experiment changing the inputs

    ### Inference
    Upload a csv file with the features to get a inference and data drift analysis
"""
)
