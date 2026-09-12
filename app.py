import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import pandas as pd
import numpy as np

from model_utils import (
    load_iris_dataset,
    check_missing_values,
    preprocess_data,
    build_fnn_model,
    train_fnn_model,
    evaluate_fnn_model,
    predict_sample,
    CLASS_NAMES,
    FEATURE_KEYS
)
from visualizations import (
    plot_training_curves,
    plot_confusion_matrix_figure,
    plot_feature_distributions,
    draw_fnn_architecture,
    plot_probability_chart
)

st.set_page_config(
    page_title="Deep FNN | Pattern Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown('''<style>
    [data-testid="stSidebar"] { display: none; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {
        background-color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0F172A;
    }
    .block-container {
        max-width: 980px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }
</style>''', unsafe_allow_html=True)
