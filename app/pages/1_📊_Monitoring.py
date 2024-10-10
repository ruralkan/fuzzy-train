import os
from pathlib import Path
import streamlit as st
from typing import Dict
from typing import List
from typing import Text

from src.ui import display_header
from src.ui import display_report
from src.ui import display_sidebar_header
from src.ui import select_period
from src.ui import select_project
from src.ui import select_report
from src.ui import set_page_container_style
from src.utils import EntityNotFoundError
from src.utils import get_reports_mapping
from src.utils import list_periods
import mlflow
import pandas as pd
import os

st.set_page_config(page_title="Monitoring", page_icon="📈")

client = mlflow.tracking.MlflowClient(tracking_uri='http://host.docker.internal:5000')
experiment_id = '5'
experiments = client.search_runs(experiment_ids=experiment_id, order_by=['metrics.mae']).to_list()
experiment_df = pd.DataFrame(experiments)
experiment_table=list()
id_table=list()
for experiment in experiments:

    raw_dict = {'run_name':experiment.to_dictionary()['info']['run_name']}
    #raw_dict.update(experiment.to_dictionary()['info']['experiment_id'])
    raw_dict.update({'run_id':experiment.to_dictionary()['info']['run_id']})
    raw_dict.update(experiment.to_dictionary()['data']['params'])
    raw_dict.update(experiment.to_dictionary()['data']['metrics'])
    #experiment_df= pd.concat([experiment_df,pd.DataFrame.from_dict(raw_dict)], axis=1)
    #experiment_df.loc[len(experiment_df)]= raw_dict
    experiment_table.append(raw_dict)
    #id_table.append(raw_dict['run_id'])
experiment_df= pd.DataFrame(experiment_table)
# Configure some styles
set_page_container_style()
# Sidebar: Logo and links
display_sidebar_header()    


st.markdown("# Reporting")

selected_project =select_project(experiment_df['run_name'].to_list())
# Display report header (UI)
display_header(
            project_name=selected_project
        )

local_dir = 'reports'
os.makedirs(local_dir, exist_ok=True)
# Display selected report(UI)
run_id = str(experiment_df[experiment_df['run_name']==selected_project]['run_id']).split(' ')[4].split('\n')[0]

st.info(run_id)
selected_report=client.download_artifacts(run_id,'data_drift_suite.html',local_dir)
report_path =Path(selected_report)
display_report(report_path)


