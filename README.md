# Deep Feedforward Neural Network for Classification

**Course:** Pattern Recognition  
**Assessment:** TAE 1: Project Based Learning – Phase I  
**Student Name:** Amogh Samarth  
**USN:** CM23034  

---

## 📑 Table of Contents
- [📖 Project Overview](#-project-overview)
- [🛠️ Technology Stack](#-technology-stack)
- [🧠 Neural Network Architecture](#-deep-feedforward-neural-network-architecture)
- [🧮 Mathematical Foundations](#-mathematical-foundations)
- [⚙️ Data Preprocessing Pipeline](#-data-preprocessing-pipeline)
- [🚀 Installation and Execution](#-installation-and-execution)
- [🖥️ Streamlit Dashboard Pages](#-streamlit-dashboard-pages)
- [🎓 Viva Questions & Answers](#-viva-questions--answers-quick-reference)
- [👨‍💻 Author Details](#-author--submission-details)

---

## 📖 Project Overview
This project presents an interactive, academic-grade web-based demonstration of a **Deep Feedforward Neural Network (FNN)** for multi-class pattern classification applied to the classic **Iris flower dataset**.

Built with **TensorFlow / Keras**, **Scikit-learn**, and **Streamlit**, this project implements an end-to-end machine learning pipeline featuring real dataset exploration, data leakage-free feature standardization, deep architectural design, mini-batch backpropagation training with EarlyStopping, comprehensive model diagnostics, and real-time inference with probability distributions.

---

## 🛠️ Technology Stack
- **Programming Language:** Python 3.10+
- **Deep Learning Framework:** TensorFlow 2.x / Keras
- **Machine Learning & Preprocessing:** Scikit-learn
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **User Interface / Web Dashboard:** Streamlit

---

## 🧠 Deep Feedforward Neural Network Architecture

The network consists of fully connected (Dense) layers:

```
[Input: 4 Features]
        ↓
[Hidden Layer 1: Dense (16 Neurons) + ReLU Activation]
        ↓
[Hidden Layer 2: Dense (16 Neurons) + ReLU Activation]
        ↓
[Hidden Layer 3: Dense (8 Neurons) + ReLU Activation]
        ↓
[Output Layer: Dense (3 Neurons) + Softmax Activation]
```

### Parameter Count Breakdown

| Layer | Type | Input Dimension | Output Dimension | Weights ($W$) | Biases ($b$) | Total Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Layer 1** | Dense + ReLU | 4 | 16 | $4 \times 16 = 64$ | 16 | **80** |
| **Layer 2** | Dense + ReLU | 16 | 16 | $16 \times 16 = 256$ | 16 | **272** |
| **Layer 3** | Dense + ReLU | 16 | 8 | $16 \times 8 = 128$ | 8 | **136** |
| **Layer 4** | Dense + Softmax | 8 | 3 | $8 \times 3 = 24$ | 3 | **27** |
| **Total** | | | | **472** | **43** | **515 Parameters** |

---

## 🧮 Mathematical Foundations

### 1. Forward Propagation Equation
For any layer $l$:
$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
Where:
- $W^{[l]}$ is the weight matrix
- $a^{[l-1]}$ is the activation output of the previous layer (with $a^{[0]} = x$)
- $b^{[l]}$ is the bias vector
- $z^{[l]}$ is the linear combination (logit)

### 2. Hidden Layer Activation (ReLU)
$$a^{[l]} = \text{ReLU}(z^{[l]}) = \max(0, z^{[l]})$$
- Mitigates the vanishing gradient problem.
- Computationally efficient piecewise-linear activation.

### 3. Output Layer Activation (Softmax)
$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{for } i \in \{1, 2, 3\}$$
- Maps arbitrary real numbers to probabilities in $[0, 1]$ such that $\sum_{i=1}^3 \sigma(z)_i = 1.0$.

### 4. Loss Function (Categorical Cross-Entropy)
$$\mathcal{L}(y, \hat{y}) = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$$
- Penalizes divergence between true one-hot vector $y$ and predicted probabilities $\hat{y}$.

---

## ⚙️ Data Preprocessing Pipeline

1. **Dataset Loading:** 150 Iris flower samples (50 Setosa, 50 Versicolor, 50 Virginica).
2. **Missing Value Check:** Verified 0 null values across all 4 features.
3. **Stratified Partition:**
   - **Training Set (70% - 104 samples):** For gradient updates.
   - **Validation Set (15% - 23 samples):** For monitoring validation loss and EarlyStopping.
   - **Test Set (15% - 23 samples):** Completely unseen holdout for evaluation.
4. **Target Encoding:** One-Hot Encoding into 3-dimensional binary vectors.
5. **StandardScaler Normalization:**
   $$z = \frac{x - \mu}{\sigma}$$
   *Crucial:* Fitted strictly on the training set to prevent data leakage.

---

## 🚀 Installation and Execution

### Step 1: Clone or Navigate to Directory
```bash
cd e:\CM23034_PR_TAE_1
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Application
```bash
streamlit run app.py
```
The interactive dashboard will automatically open in your default browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Dashboard Pages

1. **🏠 Home:** Student details, FNN definition, key principles, end-to-end deep learning workflow.
2. **📊 Dataset:** Dataset overview, tabular explorer, statistical metrics, feature distribution plots.
3. **⚙️ Preprocessing:** Pipeline flowchart, missing value audit, StandardScaler formulas, raw vs scaled comparisons.
4. **🧠 FNN Architecture:** Interactive visual network topology diagram, mathematical derivations, layer parameter calculations.
5. **🚀 Training:** Hyperparameter selectors (Epochs, Batch size, Learning rate), live training button, training/val loss & accuracy curves.
6. **📈 Evaluation:** Test accuracy, test loss, confusion matrix heatmap, classification report (Precision, Recall, F1-Score), actual vs predicted table.
7. **🔮 Interactive Prediction:** Sliders and presets for Iris dimensions, live forward pass inference, confidence percentage badge, and Softmax probability distribution bars.

---

## 🎓 Viva Questions & Answers (Quick Reference)

### Q1: Why is it called a "Feedforward" Neural Network?
**Answer:** In an FNN, signals propagate strictly forward from input to output layers without any cycles or feedback loops. Information moves unidirectionally ($x \to h_1 \to h_2 \to h_3 \to \hat{y}$).

### Q2: Why is Softmax used in the output layer instead of Sigmoid?
**Answer:** Softmax is a generalization of Sigmoid for multi-class classification ($K \ge 3$). It guarantees that all class probabilities sum to exactly $1.0$, allowing them to be interpreted as mutually exclusive class probabilities.

### Q3: Why is Categorical Cross-Entropy preferred over Mean Squared Error (MSE)?
**Answer:** Cross-Entropy is derived from maximum likelihood estimation for multinomial distributions. When paired with Softmax, its gradient is linear $(\hat{y} - y)$, preventing gradient saturation and enabling faster, more stable convergence during backpropagation.

### Q4: Why must `StandardScaler` be fitted ONLY on the training data?
**Answer:** Fitting a scaler on the entire dataset introduces **data leakage** because information about the mean and variance of the test set bleeds into the training process, giving overly optimistic performance estimates.

### Q5: What is the purpose of EarlyStopping?
**Answer:** EarlyStopping monitors the validation loss across epochs. When validation loss stops decreasing for a specified number of epochs (`patience`), training is terminated and the weights with the lowest validation loss are restored, preventing overfitting.

### Q6: How many trainable parameters are in this network?
**Answer:** Exactly **515 parameters**:
- Hidden 1: $(4 \times 16) + 16 = 80$
- Hidden 2: $(16 \times 16) + 16 = 272$
- Hidden 3: $(16 \times 8) + 8 = 136$
- Output: $(8 \times 3) + 3 = 27$
- Total: $80 + 272 + 136 + 27 = 515$.

---

## 👨‍💻 Author & Submission Details
- **Student Name:** Amogh Samarth
- **USN:** CM23034
- **Course:** Pattern Recognition
- **Institution Assignment:** TAE 1: Project Based Learning – Phase I
