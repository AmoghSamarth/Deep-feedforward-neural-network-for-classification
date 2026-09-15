"""
Standalone Offline Training & Artifact Export Script for Deep FNN
Course: Pattern Recognition (TAE 1: Project Based Learning – Phase I)
Student: Amogh Samarth | USN: CM23034

This script:
1. Loads the Iris dataset.
2. Performs stratified 70/15/15 train/val/test split.
3. Fits StandardScaler strictly on training set.
4. Builds the 4-layer Deep Feedforward Neural Network (515 parameters).
5. Trains the model with Adam and EarlyStopping.
6. Evaluates the model on unseen test data.
7. Exports real, un-fabricated weights, biases, metrics, and dataset to public/
   for production inference in Next.js on Vercel.
"""

import os
import json
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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

CLASS_NAMES = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
FEATURE_DISPLAY = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']

def main():
    print("=== STEP 1: Loading Iris Dataset ===")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=FEATURE_KEYS)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    
    # Check missing values
    null_counts = df.isnull().sum()
    missing_summary = [
        {"column": col, "missingCount": int(null_counts[col]), "missingPercentage": float(null_counts[col] / len(df) * 100)}
        for col in df.columns
    ]

    print(f"Loaded {len(df)} samples across 3 classes. Missing values: {null_counts.sum()}")

    print("\n=== STEP 2: Preprocessing & Stratified Splitting ===")
    X = df[FEATURE_KEYS].values
    y = df['target'].values

    # 15% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )

    # 15% Validation (relative to remaining 85% -> 0.15 / 0.85)
    rel_val_size = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=rel_val_size, stratify=y_train_val, random_state=RANDOM_STATE
    )

    print(f"Splits -> Train: {len(X_train)} (70%), Val: {len(X_val)} (15%), Test: {len(X_test)} (15%)")

    # One-Hot Encoding
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_val_cat = to_categorical(y_val, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)

    # Feature Scaling: Fit strictly on Train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    scaler_mean = scaler.mean_.tolist()
    scaler_scale = scaler.scale_.tolist()
    scaler_var = scaler.var_.tolist()

    print("StandardScaler fitted on training data.")
    print("Mean:", scaler_mean)
    print("Scale:", scaler_scale)

    print("\n=== STEP 3: Building FNN Model Architecture ===")
    model = Sequential(name="Deep_FNN_Classifier")
    model.add(Input(shape=(4,), name="Input_Layer"))
    model.add(Dense(16, activation='relu', name="Dense_Hidden_1_16"))
    model.add(Dense(16, activation='relu', name="Dense_Hidden_2_16"))
    model.add(Dense(8, activation='relu', name="Dense_Hidden_3_8"))
    model.add(Dense(3, activation='softmax', name="Dense_Output_3_Softmax"))

    optimizer = Adam(learning_rate=0.01)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    total_params = model.count_params()
    print(f"Built FNN model. Total trainable parameters: {total_params} (Expected: 515)")
    assert total_params == 515, f"Parameter count mismatch: {total_params} != 515"

    print("\n=== STEP 4: Training Model with EarlyStopping ===")
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train_scaled, y_train_cat,
        validation_data=(X_val_scaled, y_val_cat),
        epochs=80,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    epochs_trained = len(history.history['loss'])
    print(f"Training completed in {epochs_trained} epochs.")

    print("\n=== STEP 5: Evaluating Model on Test Set ===")
    eval_res = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
    test_loss = float(eval_res[0])
    test_accuracy = float(eval_res[1])

    y_pred_probs = model.predict(X_test_scaled, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_test, y_pred_classes).tolist()
    report = classification_report(
        y_test, y_pred_classes,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    test_samples_comparison = []
    for i in range(len(y_test)):
        actual_idx = int(y_test[i])
        pred_idx = int(y_pred_classes[i])
        conf = float(y_pred_probs[i][pred_idx])
        test_samples_comparison.append({
            "sampleIndex": i + 1,
            "rawFeatures": [round(float(v), 2) for v in X_test[i]],
            "actualClass": CLASS_NAMES[actual_idx],
            "actualIndex": actual_idx,
            "predictedClass": CLASS_NAMES[pred_idx],
            "predictedIndex": pred_idx,
            "confidence": round(conf * 100, 2),
            "match": actual_idx == pred_idx,
            "probabilities": [round(float(p), 4) for p in y_pred_probs[i]]
        })

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%, Test Loss: {test_loss:.4f}")

    print("\n=== STEP 6: Extracting Weights and Biases for Inference ===")
    weights_and_biases = []
    for layer in model.layers:
        w, b = layer.get_weights()
        weights_and_biases.append({
            "name": layer.name,
            "weights": w.tolist(),  # shape [input_dim, output_dim]
            "biases": b.tolist(),   # shape [output_dim]
            "activation": "relu" if "Hidden" in layer.name else "softmax",
            "inputDim": w.shape[0],
            "outputDim": w.shape[1],
            "paramCount": int(w.size + b.size)
        })

    model_artifact = {
        "metadata": {
            "modelName": "Deep Feedforward Neural Network (FNN)",
            "course": "Pattern Recognition (TAE 1: Phase I)",
            "student": "Amogh Samarth",
            "usn": "CM23034",
            "dateTrained": "2026-09-15",
            "totalParameters": total_params,
            "layers": [
                {"name": "Input Layer", "units": 4, "activation": "None"},
                {"name": "Hidden Layer 1", "units": 16, "activation": "ReLU", "params": 80},
                {"name": "Hidden Layer 2", "units": 16, "activation": "ReLU", "params": 272},
                {"name": "Hidden Layer 3", "units": 8, "activation": "ReLU", "params": 136},
                {"name": "Output Layer", "units": 3, "activation": "Softmax", "params": 27}
            ]
        },
        "scaler": {
            "mean": scaler_mean,
            "scale": scaler_scale,
            "var": scaler_var,
            "featureNames": FEATURE_DISPLAY,
            "featureKeys": FEATURE_KEYS
        },
        "classNames": CLASS_NAMES,
        "layers": weights_and_biases
    }

    print("\n=== STEP 7: Exporting to public/ Directories ===")
    os.makedirs("public/model", exist_ok=True)
    os.makedirs("public/data", exist_ok=True)

    with open("public/model/model.json", "w") as f:
        json.dump(model_artifact, f, indent=2)

    # Dataset export (150 samples)
    dataset_records = []
    for idx, row in df.iterrows():
        dataset_records.append({
            "id": int(idx + 1),
            "sepalLength": float(row["sepal_length"]),
            "sepalWidth": float(row["sepal_width"]),
            "petalLength": float(row["petal_length"]),
            "petalWidth": float(row["petal_width"]),
            "target": int(row["target"]),
            "species": str(row["species"])
        })

    with open("public/data/dataset.json", "w") as f:
        json.dump(dataset_records, f, indent=2)

    # Preprocessing export
    raw_sample_4 = X_train[:4].tolist()
    scaled_sample_4 = X_train_scaled[:4].tolist()
    preprocessing_artifact = {
        "missingValues": missing_summary,
        "splitCounts": {
            "total": 150,
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
            "trainPercentage": 70,
            "valPercentage": 15,
            "testPercentage": 15
        },
        "scalerStats": {
            feature: {
                "mean": round(scaler_mean[i], 4),
                "scale": round(scaler_scale[i], 4),
                "var": round(scaler_var[i], 4)
            }
            for i, feature in enumerate(FEATURE_DISPLAY)
        },
        "beforeScalingSample": raw_sample_4,
        "afterScalingSample": [[round(val, 4) for val in row] for row in scaled_sample_4]
    }

    with open("public/data/preprocessing.json", "w") as f:
        json.dump(preprocessing_artifact, f, indent=2)

    # Training history export
    training_history_artifact = {
        "epochsTrained": epochs_trained,
        "earlyStoppingRestoredEpoch": epochs_trained - 15 if epochs_trained > 15 else epochs_trained,
        "hyperparameters": {
            "optimizer": "Adam",
            "learningRate": 0.01,
            "batchSize": 16,
            "maxEpochs": 80,
            "earlyStoppingPatience": 15,
            "lossFunction": "Categorical Cross-Entropy"
        },
        "history": {
            "epoch": list(range(1, epochs_trained + 1)),
            "loss": [round(float(v), 5) for v in history.history['loss']],
            "val_loss": [round(float(v), 5) for v in history.history['val_loss']],
            "accuracy": [round(float(v) * 100, 2) for v in history.history['accuracy']],
            "val_accuracy": [round(float(v) * 100, 2) for v in history.history['val_accuracy']]
        }
    }

    with open("public/data/training_history.json", "w") as f:
        json.dump(training_history_artifact, f, indent=2)

    # Evaluation export
    evaluation_artifact = {
        "testLoss": round(test_loss, 4),
        "testAccuracy": round(test_accuracy * 100, 2),
        "testSamplesCount": len(X_test),
        "confusionMatrix": cm,
        "classNames": CLASS_NAMES,
        "classificationReport": report,
        "testSamples": test_samples_comparison
    }

    with open("public/data/evaluation.json", "w") as f:
        json.dump(evaluation_artifact, f, indent=2)

    print("\n[SUCCESS] Successfully trained model and exported all production artifacts to public/!")

if __name__ == '__main__':
    main()
