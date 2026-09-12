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

@st.cache_data
def get_dataset():
    df, _ = load_iris_dataset()
    missing_df = check_missing_values(df)
    return df, missing_df

@st.cache_resource
def get_trained_model():
    df, _ = get_dataset()
    data = preprocess_data(df)
    model = build_fnn_model(learning_rate=0.01)
    model, history = train_fnn_model(
        model,
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        epochs=80,
        batch_size=16
    )
    eval_results = evaluate_fnn_model(
        model,
        data['X_test'],
        data['y_test'],
        data['y_test_int']
    )
    return {
        'model': model,
        'history': history,
        'eval_results': eval_results,
        'data': data
    }

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

if 'model_state' not in st.session_state:
    with st.spinner("Initializing Deep Neural Network..."):
        st.session_state.model_state = get_trained_model()

model_state = st.session_state.model_state
df, missing_df = get_dataset()

nav_items = ["Home", "Dataset", "Preprocessing", "Architecture", "Training", "Evaluation", "Predict"]
cols = st.columns(len(nav_items))

for idx, item in enumerate(nav_items):
    is_active = (st.session_state.current_page == item)
    btn_type = "primary" if is_active else "secondary"
    if cols[idx].button(item, key=f"nav_{item}", use_container_width=True, type=btn_type):
        st.session_state.current_page = item
        st.rerun()

st.write("")

if st.session_state.current_page == "Home":
    st.markdown("## Deep Feedforward Neural Network for Classification")
    st.caption("Pattern Recognition • TAE 1 • Project Based Learning – Phase I")
    st.write("A Deep Feedforward Neural Network processes data in one direction through fully connected layers to learn patterns and classify inputs.")
