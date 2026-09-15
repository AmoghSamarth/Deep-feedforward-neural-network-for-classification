"""
Deep Feedforward Neural Network for Classification
Course: Pattern Recognition (TAE 1: Project Based Learning – Phase I)
Student: Amogh Samarth | USN: CM23034
Tech Stack: Python, Streamlit, TensorFlow/Keras, Scikit-learn
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import pandas as pd
import numpy as np

# Internal modular imports
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

# Page configuration
st.set_page_config(
    page_title="Deep FNN | Pattern Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# High-End Minimalist Modern CSS
st.markdown("""
<style>
    /* Hide Streamlit default chrome & sidebars */
    [data-testid="stSidebar"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    header { visibility: hidden !important; }
    .stDeployButton { display: none !important; }
    
    /* Clean base page styling */
    .stApp {
        background-color: #F8FAFC !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
        color: #0F172A !important;
    }

    /* Container constraints */
    .block-container {
        max-width: 1040px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    /* Modern Styled Tab Bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        background-color: #FFFFFF !important;
        padding: 6px 8px !important;
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        margin-bottom: 26px !important;
        justify-content: center !important;
        display: flex !important;
        flex-wrap: wrap !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px !important;
        padding: 0px 20px !important;
        background-color: transparent !important;
        border-radius: 8px !important;
        color: #64748B !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        border: 1px solid transparent !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF !important;
        color: #2563EB !important;
        border: 1px solid #BFDBFE !important;
        box-shadow: 0 1px 2px rgba(37, 99, 235, 0.08) !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Standard Button Normalization */
    div.stButton > button {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        padding: 8px 16px !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    }
    div.stButton > button:hover {
        background-color: #F8FAFC !important;
        border-color: #2563EB !important;
        color: #2563EB !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #2563EB !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
        color: #FFFFFF !important;
    }

    /* Clean Card Containers */
    .clean-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        margin-bottom: 16px;
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .clean-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .clean-card-compact {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        text-align: center;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .clean-card-compact:hover {
        border-color: #94A3B8;
        transform: translateY(-1px);
    }

    /* Typography */
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.6px;
        line-height: 1.25;
        margin-bottom: 4px;
    }
    .main-subtitle {
        font-size: 13px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 22px;
    }
    .hero-lead {
        font-size: 18px;
        font-weight: 600;
        color: #2563EB;
        margin-bottom: 8px;
    }
    .hero-text {
        font-size: 15px;
        color: #475569;
        line-height: 1.6;
        margin-bottom: 26px;
    }

    /* Metric numbers */
    .metric-title {
        font-size: 11px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-sub {
        font-size: 12px;
        color: #64748B;
        margin-top: 2px;
    }

    /* Flow step pill */
    .flow-step {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 14px;
        text-align: center;
        font-size: 13px;
        font-weight: 600;
        color: #1E293B;
    }
    .flow-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #94A3B8;
        font-size: 16px;
        font-weight: bold;
    }
    
    /* Result Badge */
    .result-badge {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1D4ED8;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 16px;
    }
    .result-badge-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #3B82F6;
        margin-bottom: 4px;
    }
    .result-badge-title {
        font-size: 26px;
        font-weight: 700;
        color: #1E3A8A;
    }

    /* Footer */
    .minimal-footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        font-size: 12.5px;
        color: #64748B;
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)


# Application State Caching
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


if 'model_state' not in st.session_state:
    with st.spinner("Initializing Deep Neural Network..."):
        st.session_state.model_state = get_trained_model()

model_state = st.session_state.model_state
df, missing_df = get_dataset()


# ==============================================================================
# TOP TABS NAVIGATION (Instant, Reliable, Legible, Beautiful)
# ==============================================================================
tab_home, tab_dataset, tab_prep, tab_arch, tab_train, tab_eval, tab_predict = st.tabs([
    "Home", "Dataset", "Preprocessing", "Architecture", "Training", "Evaluation", "Predict"
])


# ==============================================================================
# 1. HOME TAB
# ==============================================================================
with tab_home:
    st.markdown('<div class="main-title">Deep Feedforward Neural Network<br>for Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Pattern Recognition • TAE 1 • Project Based Learning – Phase I</div>', unsafe_allow_html=True)

    st.markdown('<div class="hero-lead">From input features to intelligent classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-text">A Deep Feedforward Neural Network processes data in one direction through fully connected layers to learn patterns and classify inputs.</div>', unsafe_allow_html=True)

    # 3 Small Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="clean-card-compact">
            <div class="metric-title">Dataset</div>
            <div class="metric-value" style="font-size:20px;">Iris Dataset</div>
            <div class="metric-sub">150 Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="clean-card-compact">
            <div class="metric-title">Model</div>
            <div class="metric-value" style="font-size:20px;">Deep FNN</div>
            <div class="metric-sub">3 Classes</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="clean-card-compact">
            <div class="metric-title">Algorithm</div>
            <div class="metric-value" style="font-size:20px;">Adam + ReLU</div>
            <div class="metric-sub">Softmax Output</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Simple Visual Workflow (Clean & Horizontal on Desktop)
    st.markdown('<div class="metric-title" style="margin-bottom:12px;">Machine Learning Pipeline</div>', unsafe_allow_html=True)
    w1, a1, w2, a2, w3, a3, w4, a4, w5, a5, w6 = st.columns([1.2, 0.2, 1.4, 0.2, 1.4, 0.2, 1.2, 0.2, 1.2, 0.2, 1.2])
    
    with w1:
        st.markdown('<div class="flow-step">Dataset</div>', unsafe_allow_html=True)
    with a1:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    with w2:
        st.markdown('<div class="flow-step">Preprocessing</div>', unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    with w3:
        st.markdown('<div class="flow-step">Neural Network</div>', unsafe_allow_html=True)
    with a3:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    with w4:
        st.markdown('<div class="flow-step">Training</div>', unsafe_allow_html=True)
    with a4:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    with w5:
        st.markdown('<div class="flow-step">Evaluation</div>', unsafe_allow_html=True)
    with a5:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    with w6:
        st.markdown('<div class="flow-step" style="border-color:#2563EB; color:#2563EB;">Prediction</div>', unsafe_allow_html=True)

    # Optional Collapsible Technical Details
    with st.expander("View Technical Details"):
        st.markdown(r"""
        - **Unidirectional Flow:** Signals propagate strictly forward: $x \to h^{(1)} \to h^{(2)} \to h^{(3)} \to \hat{y}$.
        - **Optimization:** Mini-batch gradient descent using Adam with early stopping to prevent overfitting.
        - **Loss Objective:** Categorical Cross-Entropy across 3 one-hot encoded flower species.
        """)

    # Bottom Credits
    st.markdown("""
    <div class="minimal-footer">
        <b>Amogh Samarth</b> &nbsp;•&nbsp; <b>CM23034</b><br>
        Pattern Recognition &nbsp;•&nbsp; TAE 1 Phase I
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 2. DATASET TAB
# ==============================================================================
with tab_dataset:
    st.markdown('<div class="main-title">Iris Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Benchmark Multi-Class Pattern Recognition Data</div>', unsafe_allow_html=True)

    # 4 Small Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="clean-card-compact">
            <div class="metric-title">Total Samples</div>
            <div class="metric-value">150</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="clean-card-compact">
            <div class="metric-title">Features</div>
            <div class="metric-value">4</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="clean-card-compact">
            <div class="metric-title">Classes</div>
            <div class="metric-value">3</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="clean-card-compact">
            <div class="metric-title">Missing Values</div>
            <div class="metric-value">0</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Clean DataFrame Preview
    st.markdown('<div class="metric-title">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(6), use_container_width=True)

    # Simple Feature Visualization
    st.write("")
    st.markdown('<div class="metric-title">Feature Distributions by Class</div>', unsafe_allow_html=True)
    fig_dist = plot_feature_distributions(df)
    st.pyplot(fig_dist)

    # About the Dataset (2-3 sentences)
    st.markdown("""
    <div class="clean-card" style="margin-top:16px;">
        <div class="metric-title" style="margin-bottom:6px;">About the Dataset</div>
        <div style="font-size:14px; color:#475569; line-height:1.6;">
            The Iris dataset contains morphological measurements in centimeters for three flower species: Setosa, Versicolor, and Virginica. 
            Each sample includes sepal length, sepal width, petal length, and petal width. 
            It is a balanced dataset with exactly 50 samples per class, making it ideal for multi-class classification.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 3. PREPROCESSING TAB
# ==============================================================================
with tab_prep:
    st.markdown('<div class="main-title">Data Preprocessing</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Feature Scaling & Pipeline Flow</div>', unsafe_allow_html=True)

    # Pipeline as connected cards
    st.markdown('<div class="metric-title" style="margin-bottom:12px;">Preprocessing Pipeline</div>', unsafe_allow_html=True)
    p1, ap1, p2, ap2, p3, ap3, p4, ap4, p5 = st.columns([1.2, 0.2, 1.4, 0.2, 1.4, 0.2, 1.4, 0.2, 1.5])
    with p1:
        st.markdown('<div class="flow-step">Raw Data</div>', unsafe_allow_html=True)
    with ap1:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="flow-step">Missing Check</div>', unsafe_allow_html=True)
    with ap2:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    with p3:
        st.markdown('<div class="flow-step">Feature Scaling</div>', unsafe_allow_html=True)
    with ap3:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    with p4:
        st.markdown('<div class="flow-step">Target Encoding</div>', unsafe_allow_html=True)
    with ap4:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    with p5:
        st.markdown('<div class="flow-step" style="border-color:#2563EB; color:#2563EB;">Train / Val / Test</div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Before Scaling | After Scaling
    st.markdown('<div class="metric-title">Before Scaling vs. After Scaling</div>', unsafe_allow_html=True)
    
    data = model_state['data']
    raw_sample = data['X_train_raw'][:4]
    scaled_sample = data['X_train'][:4]

    col_b, col_a = st.columns(2)
    with col_b:
        df_before = pd.DataFrame(raw_sample, columns=['Sepal L', 'Sepal W', 'Petal L', 'Petal W'])
        st.markdown('<div style="font-size:13px; font-weight:600; color:#64748B; margin-bottom:6px;">Before Scaling (Raw cm)</div>', unsafe_allow_html=True)
        st.dataframe(df_before.round(1), use_container_width=True)

    with col_a:
        df_after = pd.DataFrame(scaled_sample, columns=['Sepal L', 'Sepal W', 'Petal L', 'Petal W'])
        st.markdown('<div style="font-size:13px; font-weight:600; color:#2563EB; margin-bottom:6px;">After Scaling (StandardScaler)</div>', unsafe_allow_html=True)
        st.dataframe(df_after.round(3), use_container_width=True)

    # Short Mathematical Explanation
    st.markdown("""
    <div class="clean-card" style="margin-top:10px;">
        <div class="metric-title" style="margin-bottom:8px;">Standardization Formula</div>
        <div style="font-size:16px; font-weight:600; color:#0F172A; margin-bottom:6px;">
            z = (x - &mu;) / &sigma;
        </div>
        <div style="font-size:13.5px; color:#64748B; line-height:1.5;">
            StandardScaler shifts feature values to zero mean (&mu; = 0) and unit variance (&sigma; = 1). 
            Crucially, the scaler is fitted strictly on the training set to prevent data leakage into validation and testing.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("View Technical Details"):
        st.markdown("""
        - **Data Split:** Stratified 70% Training (104 samples), 15% Validation (23 samples), 15% Holdout Test (23 samples).
        - **Target Encoding:** One-Hot Encoded vectors: Setosa `[1, 0, 0]`, Versicolor `[0, 1, 0]`, Virginica `[0, 0, 1]`.
        """)


# ==============================================================================
# 4. ARCHITECTURE TAB
# ==============================================================================
with tab_arch:
    st.markdown('<div class="main-title">FNN Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Fully Connected Layer Topology</div>', unsafe_allow_html=True)

    # Large Clean Neural Network Diagram
    fig_arch = draw_fnn_architecture()
    st.pyplot(fig_arch)

    # Layer Flow Label
    st.markdown("""
    <div style="text-align:center; font-size:13px; font-weight:600; color:#64748B; margin-top:8px; margin-bottom:20px;">
        Input Layer (4) &nbsp;→&nbsp; Hidden Layers (16 → 16 → 8) &nbsp;→&nbsp; Output Layer (3)
    </div>
    """, unsafe_allow_html=True)

    # Explanations & Math Card
    col_exp, col_eq = st.columns([1.5, 1])
    with col_exp:
        st.markdown("""
        <div class="clean-card" style="height:100%;">
            <div class="metric-title" style="margin-bottom:8px;">Activation Functions</div>
            <div style="font-size:14px; color:#1E293B; line-height:1.6;">
                • <b>ReLU</b> helps the network learn non-linear patterns.<br>
                • <b>Softmax</b> converts the final outputs into class probabilities summing to 1.0.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_eq:
        st.markdown("""
        <div class="clean-card" style="height:100%; text-align:center;">
            <div class="metric-title" style="margin-bottom:8px;">Forward Propagation</div>
            <div style="font-size:22px; font-weight:700; color:#2563EB; margin-top:4px;">
                z = Wa + b
            </div>
            <div style="font-size:12px; color:#64748B; margin-top:4px;">Linear combination before activation</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("View Technical Details"):
        st.markdown("""
        **Parameter Calculation Breakdown:**
        - Hidden 1: (4 × 16) + 16 = 80 params
        - Hidden 2: (16 × 16) + 16 = 272 params
        - Hidden 3: (16 × 8) + 8 = 136 params
        - Output: (8 × 3) + 3 = 27 params
        - **Total Trainable Parameters:** 515
        """)


# ==============================================================================
# 5. TRAINING TAB
# ==============================================================================
with tab_train:
    st.markdown('<div class="main-title">Model Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Convergence & Accuracy Progression</div>', unsafe_allow_html=True)

    history = model_state['history']
    final_epochs = len(history.history['loss'])
    train_acc = history.history['accuracy'][-1] * 100
    val_acc = history.history['val_accuracy'][-1] * 100

    # 3 Metrics
    tm1, tm2, tm3 = st.columns(3)
    with tm1:
        st.markdown(f"""
        <div class="clean-card-compact">
            <div class="metric-title">Epochs Run</div>
            <div class="metric-value">{final_epochs}</div>
        </div>
        """, unsafe_allow_html=True)
    with tm2:
        st.markdown(f"""
        <div class="clean-card-compact">
            <div class="metric-title">Training Accuracy</div>
            <div class="metric-value">{train_acc:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with tm3:
        st.markdown(f"""
        <div class="clean-card-compact">
            <div class="metric-title">Validation Accuracy</div>
            <div class="metric-value">{val_acc:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Two Clean Charts: Accuracy & Loss
    fig_curves = plot_training_curves(history)
    st.pyplot(fig_curves)

    # Clean Configuration Card Below Charts
    st.markdown("""
    <div class="clean-card" style="margin-top:16px;">
        <div class="metric-title" style="margin-bottom:8px;">Training Configuration</div>
        <div style="display:flex; justify-content:space-between; flex-wrap:wrap; font-size:13.5px; color:#475569;">
            <span><b>Optimizer:</b> Adam</span>
            <span><b>Loss:</b> Categorical Cross-Entropy</span>
            <span><b>Activation:</b> ReLU + Softmax</span>
            <span><b>Early Stopping:</b> Enabled (Patience: 15)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("View Technical Details"):
        if st.button("Retrain Model Live", key="retrain_btn"):
            with st.spinner("Retraining FNN with fresh random initialization..."):
                data = model_state['data']
                new_model = build_fnn_model(learning_rate=0.01)
                new_model, new_hist = train_fnn_model(
                    new_model,
                    data['X_train'], data['y_train'],
                    data['X_val'], data['y_val'],
                    epochs=80, batch_size=16
                )
                eval_res = evaluate_fnn_model(new_model, data['X_test'], data['y_test'], data['y_test_int'])
                st.session_state.model_state = {
                    'model': new_model,
                    'history': new_hist,
                    'eval_results': eval_res,
                    'data': data
                }
                st.success("Model retrained successfully!")
                st.rerun()


# ==============================================================================
# 6. EVALUATION TAB
# ==============================================================================
with tab_eval:
    st.markdown('<div class="main-title">Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Independent Test Set Verification</div>', unsafe_allow_html=True)

    eval_results = model_state['eval_results']
    test_acc = eval_results['test_accuracy'] * 100

    # Prominent Test Accuracy
    st.markdown(f"""
    <div class="clean-card" style="text-align:center; padding:24px 16px; margin-bottom:20px;">
        <div class="metric-title">Test Accuracy</div>
        <div style="font-size:42px; font-weight:800; color:#2563EB; letter-spacing:-1px;">
            {test_acc:.1f}%
        </div>
        <div style="font-size:13px; color:#64748B; margin-top:2px;">Evaluated on 23 holdout samples unseen during training</div>
    </div>
    """, unsafe_allow_html=True)

    # Confusion Matrix & Classification Report side-by-side
    col_cm, col_cr = st.columns([1, 1.1])
    with col_cm:
        st.markdown('<div class="metric-title">Confusion Matrix</div>', unsafe_allow_html=True)
        fig_cm = plot_confusion_matrix_figure(eval_results['confusion_matrix'], CLASS_NAMES)
        st.pyplot(fig_cm)

    with col_cr:
        st.markdown('<div class="metric-title">Classification Report</div>', unsafe_allow_html=True)
        rep_df = pd.DataFrame(eval_results['report_dict']).T
        rep_df = rep_df.round(2)
        st.dataframe(rep_df[['precision', 'recall', 'f1-score', 'support']], use_container_width=True)

    st.write("")

    # Actual vs Predicted (Small Compact Table)
    st.markdown('<div class="metric-title">Actual vs. Predicted</div>', unsafe_allow_html=True)
    st.dataframe(eval_results['comparison_df'].head(8), use_container_width=True)


# ==============================================================================
# 7. PREDICTION TAB
# ==============================================================================
with tab_predict:
    st.markdown('<div class="main-title">Try the Model</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:15px; color:#64748B; margin-bottom:20px;">Enter flower measurements and let the neural network classify the flower.</div>', unsafe_allow_html=True)

    # State variables for inputs
    if 'pred_sl' not in st.session_state:
        st.session_state.pred_sl = 6.4
        st.session_state.pred_sw = 3.0
        st.session_state.pred_pl = 5.3
        st.session_state.pred_pw = 2.0

    # Preset helper
    p1, p2, p3 = st.columns(3)
    if p1.button("Sample: Setosa", key="btn_setosa", use_container_width=True):
        st.session_state.pred_sl, st.session_state.pred_sw, st.session_state.pred_pl, st.session_state.pred_pw = 5.0, 3.5, 1.4, 0.2
        st.rerun()
    if p2.button("Sample: Versicolor", key="btn_versicolor", use_container_width=True):
        st.session_state.pred_sl, st.session_state.pred_sw, st.session_state.pred_pl, st.session_state.pred_pw = 6.0, 2.9, 4.5, 1.3
        st.rerun()
    if p3.button("Sample: Virginica", key="btn_virginica", use_container_width=True):
        st.session_state.pred_sl, st.session_state.pred_sw, st.session_state.pred_pl, st.session_state.pred_pw = 6.7, 3.1, 5.6, 2.4
        st.rerun()

    st.write("")

    # Clean Input Card with 4 inputs
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    c_i1, c_i2 = st.columns(2)
    with c_i1:
        sl = st.slider("Sepal Length (cm)", 4.0, 8.0, float(st.session_state.pred_sl), 0.1, key="sl_input")
        sw = st.slider("Sepal Width (cm)", 2.0, 4.5, float(st.session_state.pred_sw), 0.1, key="sw_input")
    with c_i2:
        pl = st.slider("Petal Length (cm)", 1.0, 7.0, float(st.session_state.pred_pl), 0.1, key="pl_input")
        pw = st.slider("Petal Width (cm)", 0.1, 2.6, float(st.session_state.pred_pw), 0.1, key="pw_input")

    predict_clicked = st.button("Predict", key="predict_action", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Real Model Inference (updates dynamically or on click)
    pred = predict_sample(
        model_state['model'],
        model_state['data']['scaler'],
        sl, sw, pl, pw
    )

    # Result Card
    st.markdown(f"""
    <div class="result-badge">
        <div class="result-badge-label">Predicted Class</div>
        <div class="result-badge-title">{pred['predicted_class']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Horizontal Probability Chart
    st.markdown('<div class="metric-title" style="margin-bottom:8px;">Class Probabilities</div>', unsafe_allow_html=True)
    fig_prob = plot_probability_chart(pred['probabilities'], CLASS_NAMES)
    st.pyplot(fig_prob)

    with st.expander("View Technical Details"):
        st.markdown(f"""
        - **Raw Vector:** `[{sl}, {sw}, {pl}, {pw}]`
        - **Standardized Vector:** `{np.round(pred['scaled_input'], 3).tolist()}`
        - **Highest Softmax Confidence:** `{pred['confidence']*100:.2f}%`
        """)
