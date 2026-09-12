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
