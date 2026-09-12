import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import tensorflow as tf

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
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=FEATURE_KEYS)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    return df, iris

def check_missing_values(df):
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Feature / Column': df.columns,
        'Missing Count': null_counts.values,
        'Missing Percentage (%)': null_percentages.values
    })
    return missing_summary

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

def preprocess_data(df, test_size=0.15, val_size=0.15, random_state=RANDOM_STATE):
    X = df[FEATURE_KEYS].values
    y = df['target'].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=random_state
    )

    y_train_cat = to_categorical(y_train, num_classes=3)
    y_val_cat = to_categorical(y_val, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)

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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def build_fnn_model(learning_rate=0.01):
    model = Sequential(name="Deep_FNN_Classifier")
    model.add(Input(shape=(4,), name="Input_Layer"))
    model.add(Dense(16, activation='relu', name="Dense_Hidden_1_16"))
    model.add(Dense(16, activation='relu', name="Dense_Hidden_2_16"))
    model.add(Dense(8, activation='relu', name="Dense_Hidden_3_8"))
    model.add(Dense(3, activation='softmax', name="Dense_Output_3_Softmax"))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_fnn_model(model, X_train, y_train, X_val, y_val, epochs=80, batch_size=16, patience=15, callbacks=None):
    if callbacks is None:
        callbacks = []
    
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

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def evaluate_fnn_model(model, X_test, y_test_cat, y_test_int, class_names=CLASS_NAMES):
    eval_results = model.evaluate(X_test, y_test_cat, verbose=0)
    test_loss = eval_results[0]
    test_accuracy = eval_results[1]

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_test_int, y_pred_classes)

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
