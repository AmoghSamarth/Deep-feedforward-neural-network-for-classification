"""
Model Utilities for Deep Feedforward Neural Network (FNN) Classification
Course: Pattern Recognition (TAE 1: Project Based Learning – Phase I)
Student: Amogh Samarth | USN: CM23034

This module handles:
1. Iris Dataset Loading & Exploration
2. Preprocessing: StandardScaler normalization, Stratified Train/Val/Test split, One-Hot Encoding
3. TensorFlow/Keras FNN architecture construction
4. Model Compilation & Training with EarlyStopping
5. Model Evaluation (Accuracy, Confusion Matrix, Classification Report)
6. Real-time inference / prediction with probability distribution
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Fixed random seed for reproducibility across runs
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

CLASS_NAMES = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
FEATURE_NAMES = [
    'Sepal Length (cm)',
    'Sepal Width (cm)',
    'Petal Length (cm)',
    'Petal Width (cm)'
]
FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']


def load_iris_dataset():
    """
    Loads the classic Iris flower dataset from scikit-learn.
    Returns:
        df (pd.DataFrame): Clean DataFrame containing features, target code, and species name.
        raw_bunch: Original sklearn dataset bunch for raw metadata access.
    """
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=FEATURE_KEYS)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    return df, iris


def check_missing_values(df):
    """
    Checks for null or NaN values in the dataset.
    Returns:
        missing_summary (pd.DataFrame): Null count and percentage per column.
    """
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Feature / Column': df.columns,
        'Missing Count': null_counts.values,
        'Missing Percentage (%)': null_percentages.values
    })
    return missing_summary


def preprocess_data(df, test_size=0.15, val_size=0.15, random_state=RANDOM_STATE):
    """
    Performs data preprocessing:
    1. Splits dataset into Train (70%), Validation (15%), and Test (15%) stratified by class.
    2. Converts target into One-Hot Encoded vectors (3 classes).
    3. Fits StandardScaler on Train features ONLY to avoid data leakage, then transforms Val and Test.

    Returns:
        data_dict (dict): Dictionary with all processed splits, targets, and fitted scaler.
    """
    X = df[FEATURE_KEYS].values
    y = df['target'].values

    # Step 1: First split into (Train + Val) and Test
    # Test ratio = test_size (e.g. 0.15)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Step 2: Split (Train + Val) into Train and Validation
    # Calculate relative validation size
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=random_state
    )

    # Step 3: One-Hot Encoding for categorical cross-entropy
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_val_cat = to_categorical(y_val, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)

    # Step 4: Feature Scaling using StandardScaler
    # Crucial: Fit scaler only on training set to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train_raw': X_train,
        'X_val_raw': X_val,
        'X_test_raw': X_test,
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train_int': y_train,
        'y_val_int': y_val,
        'y_test_int': y_test,
        'y_train': y_train_cat,
        'y_val': y_val_cat,
        'y_test': y_test_cat,
        'scaler': scaler
    }


def build_fnn_model(learning_rate=0.01):
    """
    Constructs the Deep Feedforward Neural Network (FNN) architecture:
    - Input: 4 Features
    - Hidden Layer 1: Dense(16, activation='relu')
    - Hidden Layer 2: Dense(16, activation='relu')
    - Hidden Layer 3: Dense(8, activation='relu')
    - Output Layer: Dense(3, activation='softmax')

    Every neuron in each dense layer connects to every neuron in the subsequent layer.
    """
    model = Sequential(name="Deep_FNN_Classifier")
    
    # Input layer specification
    model.add(Input(shape=(4,), name="Input_Layer"))
    
    # Hidden Layer 1: 16 neurons with ReLU activation
    model.add(Dense(16, activation='relu', name="Dense_Hidden_1_16"))
    
    # Hidden Layer 2: 16 neurons with ReLU activation
    model.add(Dense(16, activation='relu', name="Dense_Hidden_2_16"))
    
    # Hidden Layer 3: 8 neurons with ReLU activation
    model.add(Dense(8, activation='relu', name="Dense_Hidden_3_8"))
    
    # Output Layer: 3 neurons with Softmax activation (multi-class probabilities)
    model.add(Dense(3, activation='softmax', name="Dense_Output_3_Softmax"))

    # Compilation with Adam optimizer, Categorical Cross-Entropy loss, and Accuracy metric
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_fnn_model(model, X_train, y_train, X_val, y_val, epochs=80, batch_size=16, patience=15, callbacks=None):
    """
    Trains the FNN model using mini-batch gradient descent and early stopping.
    Returns:
        model: Trained Keras model
        history: Keras Training history dict containing loss, accuracy, val_loss, val_accuracy
    """
    if callbacks is None:
        callbacks = []
    
    # Early stopping callback to prevent overfitting and restore optimal weights
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )
    callbacks.append(early_stop)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    return model, history


def evaluate_fnn_model(model, X_test, y_test_cat, y_test_int, class_names=CLASS_NAMES):
    """
    Evaluates the trained FNN model on the unseen test dataset.
    Returns:
        eval_dict (dict): Contains loss, accuracy, predictions, confusion matrix, and classification report.
    """
    # Evaluate loss and accuracy
    eval_results = model.evaluate(X_test, y_test_cat, verbose=0)
    test_loss = eval_results[0]
    test_accuracy = eval_results[1]

    # Predict probabilities and compute class labels
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test_int, y_pred_classes)

    # Classification report as both dict and formatted string
    report_dict = classification_report(
        y_test_int, y_pred_classes,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_test_int, y_pred_classes,
        target_names=class_names,
        zero_division=0
    )

    # Actual vs Predicted comparison dataframe
    comparison_df = pd.DataFrame({
        'Sample #': [f"Test Sample {i+1}" for i in range(len(y_test_int))],
        'Actual Class': [class_names[i] for i in y_test_int],
        'Predicted Class': [class_names[i] for i in y_pred_classes],
        'Confidence': [f"{y_pred_probs[i][y_pred_classes[i]] * 100:.2f}%" for i in range(len(y_test_int))],
        'Match': ['✓ Correct' if actual == pred else '✗ Incorrect' for actual, pred in zip(y_test_int, y_pred_classes)]
    })

    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'y_pred_probs': y_pred_probs,
        'y_pred_classes': y_pred_classes,
        'confusion_matrix': cm,
        'report_dict': report_dict,
        'report_text': report_text,
        'comparison_df': comparison_df
    }


def predict_sample(model, scaler, sepal_length, sepal_width, petal_length, petal_width, class_names=CLASS_NAMES):
    """
    Executes real-time inference on user input:
    1. Standardizes features using the fitted StandardScaler.
    2. Executes forward pass through the trained FNN model.
    3. Returns the predicted class and full probability distribution.
    """
    raw_vector = np.array([[sepal_length, sepal_width, petal_length, petal_width]], dtype=np.float32)
    scaled_vector = scaler.transform(raw_vector)

    probabilities = model.predict(scaled_vector, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_idx]
    predicted_confidence = float(probabilities[predicted_idx])

    prob_breakdown = {
        class_names[i]: float(probabilities[i]) for i in range(len(class_names))
    }

    return {
        'predicted_class': predicted_class,
        'predicted_index': predicted_idx,
        'confidence': predicted_confidence,
        'probabilities': prob_breakdown,
        'raw_input': [sepal_length, sepal_width, petal_length, petal_width],
        'scaled_input': scaled_vector[0].tolist()
    }
