# Deep Feedforward Neural Network for Classification

![Next.js](https://img.shields.io/badge/Next.js-15.x-black?logo=next.js)
![React](https://img.shields.io/badge/React-19.x-blue?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue?logo=typescript)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38bdf8?logo=tailwind-css)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Vercel](https://img.shields.io/badge/Vercel-Production_Ready-black?logo=vercel)

**Course:** Pattern Recognition  
**Assessment:** TAE 1: Project Based Learning – Phase I  
**Student Name:** Amogh Samarth  
**USN:** CM23034  

---

## 📑 Table of Contents
1. [📖 Project Overview](#1-project-overview)
2. [✨ Features](#2-features)
3. [🧠 Neural Network Architecture](#3-neural-network-architecture)
4. [🧮 Mathematical Foundations](#4-mathematical-foundations)
5. [📊 Dataset](#5-dataset)
6. [⚙️ Preprocessing](#6-preprocessing)
7. [🚀 Model Training](#7-model-training)
8. [📈 Evaluation](#8-evaluation)
9. [🔮 Interactive Prediction](#9-interactive-prediction)
10. [🛠️ Technology Stack](#10-technology-stack)
11. [📁 Project Structure](#11-project-structure)
12. [💻 Local Development](#12-local-development)
13. [🚢 Production Deployment](#13-production-deployment)
14. [▲ Vercel Deployment](#14-vercel-deployment)
15. [⚡ Model Inference Architecture](#15-model-inference-architecture)
16. [🎓 Viva Questions & Answers](#16-viva-questions--answers-quick-reference)
17. [👨‍💻 Author Details](#17-author-details)

---

## 1. Project Overview
This project presents a high-performance, academic-grade web application demonstrating a **Deep Feedforward Neural Network (FNN)** applied to the benchmark **Iris flower classification dataset** for multi-class pattern recognition.

Originally prototyped with Streamlit, the application has been engineered into a production-grade **Next.js 15 (React 19, TypeScript, Tailwind CSS)** web platform optimized for deployment on **Vercel**. The application decouples offline deep learning model training (TensorFlow/Keras) from production inference, ensuring sub-millisecond client-side classification with zero serverless cold-start latency, zero Python runtime dependencies in production, and 100% preservation of the genuine 515-parameter neural network topology and real training metrics.

---

## 2. Features
- **Modern Academic Dashboard:** Clean, responsive, publication-grade user interface with custom scientific typography and color tokens.
- **Interactive Network Topology:** Interactive SVG visualization of all 5 layers with hover column highlights, node activations, and real parameter breakdown.
- **Strict Data Leakage Prevention:** `StandardScaler` feature normalization fitted exclusively on the 70% training split.
- **Genuine Offline Training Metrics:** Dual interactive charts displaying actual loss and accuracy progression curves across 80 epochs with EarlyStopping weight restoration at epoch 68.
- **Holdout Test Set Verification:** Verified 95.65% test accuracy on 23 unseen samples, complete with an interactive 3×3 confusion matrix heatmap and precision/recall/F1-score classification report.
- **Interactive Prediction Laboratory:** Real-time sliders, presets for all three Iris species, instant forward-pass inference, Softmax probability distribution bars summing to 1.0, and raw vs standardized vector inspections.
- **Searchable Dataset Explorer:** Filterable, paginated tabular explorer for all 150 Iris records with feature distribution ranges.
- **Comprehensive Academic viva Guide:** Built-in viva reference covering all theoretical and mathematical principles.

---

## 3. Neural Network Architecture

The network consists of fully connected (Dense) feedforward layers:

```
[Input: 4 Features (Sepal Length, Sepal Width, Petal Length, Petal Width)]
        ↓
[Hidden Layer 1: Dense (16 Neurons) + ReLU Activation]
        ↓
[Hidden Layer 2: Dense (16 Neurons) + ReLU Activation]
        ↓
[Hidden Layer 3: Dense (8 Neurons) + ReLU Activation]
        ↓
[Output Layer: Dense (3 Neurons) + Softmax Activation]
```

### Parameter Count Breakdown (Exactly 515 Trainable Parameters)

| Layer | Type | Input Dimension | Output Dimension | Weights ($W$) | Biases ($b$) | Total Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Layer 1** | Dense + ReLU | 4 | 16 | $4 \times 16 = 64$ | 16 | **80** |
| **Layer 2** | Dense + ReLU | 16 | 16 | $16 \times 16 = 256$ | 16 | **272** |
| **Layer 3** | Dense + ReLU | 16 | 8 | $16 \times 8 = 128$ | 8 | **136** |
| **Layer 4** | Dense + Softmax | 8 | 3 | $8 \times 3 = 24$ | 3 | **27** |
| **Total** | | | | **472** | **43** | **515 Parameters** |

Every neuron in each dense layer connects to every neuron in the subsequent layer without any backwards cycles or recurrence.

---

## 4. Mathematical Foundations

### 1. Forward Propagation Equation
For any layer $l$:
$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
Where:
- $W^{[l]}$ is the weight matrix of dimensions $(\text{units}^{[l]} \times \text{units}^{[l-1]})$
- $a^{[l-1]}$ is the activation output of the previous layer (with $a^{[0]} = x_{\text{norm}}$)
- $b^{[l]}$ is the bias vector of dimension $(\text{units}^{[l]} \times 1)$
- $z^{[l]}$ is the linear pre-activation combination (logit)

### 2. Hidden Layer Activation (ReLU)
$$a^{[l]} = \text{ReLU}(z^{[l]}) = \max(0, z^{[l]})$$
- Mitigates the vanishing gradient problem in deep networks.
- Computes derivative with maximum computational efficiency:
  $$\frac{d}{dz}\text{ReLU}(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$

### 3. Output Layer Activation (Softmax)
$$\sigma(z)_i = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{K} e^{z_j - \max(z)}} \quad \text{for } i \in \{1, 2, 3\}$$
- Maps arbitrary real logits to probabilities in $[0, 1]$ such that $\sum_{i=1}^3 \sigma(z)_i = 1.0$.
- Implemented with subtraction of $\max(z)$ for absolute numerical stability.

### 4. Loss Function (Categorical Cross-Entropy)
$$\mathcal{L}(y, \hat{y}) = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$$
- Derived from maximum likelihood estimation under multinomial sampling.
- When paired with Softmax output units, the gradient with respect to logits is linear:
  $$\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i$$
  which ensures rapid, robust convergence without gradient saturation.

---

## 5. Dataset
- **Name:** Iris Flower Dataset (Fisher, 1936)
- **Total Samples:** 150 instances
- **Classes (3 balanced species, 50 samples each):**
  1. *Iris Setosa* (Target 0)
  2. *Iris Versicolor* (Target 1)
  3. *Iris Virginica* (Target 2)
- **Features (4 continuous measurements in cm):**
  1. `sepal_length`: Sepal Length (4.3 to 7.9 cm)
  2. `sepal_width`: Sepal Width (2.0 to 4.4 cm)
  3. `petal_length`: Petal Length (1.0 to 6.9 cm)
  4. `petal_width`: Petal Width (0.1 to 2.5 cm)
- **Missing Values:** Exactly 0 missing or NaN values across all 150 instances.

---

## 6. Preprocessing

1. **Missing Value Audit:** Complete inspection confirms 0 null records across all columns.
2. **Stratified Split (Random Seed 42):**
   - **Training Set (70% — 104 samples):** For gradient updates during backpropagation.
   - **Validation Set (15% — 23 samples):** For monitoring validation loss and triggering EarlyStopping.
   - **Holdout Test Set (15% — 23 samples):** Completely unseen partition for unbiased final evaluation.
3. **Target One-Hot Encoding:**
   - Setosa: $[1, 0, 0]$
   - Versicolor: $[0, 1, 0]$
   - Virginica: $[0, 0, 1]$
4. **StandardScaler Feature Normalization:**
   $$z = \frac{x - \mu}{\sigma}$$
   > [!IMPORTANT]
   > **Data Leakage Prevention:** The mean ($\mu_{\text{train}}$) and scale ($\sigma_{\text{train}}$) parameters are calculated strictly from the 104 training instances. The validation and test sets are transformed using these training parameters without recalculation, guaranteeing zero data leakage.

---

## 7. Model Training

Training is executed offline using the dedicated script `training/train_model.py`:
- **Optimizer:** Adam (Adaptive Moment Estimation) with $\text{lr} = 0.01$, $\beta_1 = 0.9$, $\beta_2 = 0.999$.
- **Loss Function:** Categorical Cross-Entropy.
- **Batch Size:** 16 samples per mini-batch.
- **Max Epochs:** 80.
- **EarlyStopping:** Monitored `val_loss` with patience of 15 epochs and `restore_best_weights=True`.
- **Training Progression:**
  - Training completed in 80 epochs.
  - Optimal weights restored from epoch 68 (where validation loss reached minimum $0.00092$ and validation accuracy reached $100\%$).
  - Final training accuracy: $100.0\%$.
  - Final validation accuracy: $100.0\%$.

All metrics are exported to `public/data/training_history.json` and rendered in the web UI.

---

## 8. Evaluation

Evaluated on the independent 23-sample holdout test set (unseen during training):
- **Generalization Test Accuracy:** **95.65%** (22 out of 23 samples correctly classified).
- **Test Loss:** 0.2659.

### 3×3 Confusion Matrix

| Actual \ Predicted | Setosa | Versicolor | Virginica |
| :--- | :---: | :---: | :---: |
| **Iris Setosa** | **7** | 0 | 0 |
| **Iris Versicolor** | 0 | **7** | 1 |
| **Iris Virginica** | 0 | 0 | **8** |

### Classification Report

| Species | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Iris Setosa** | 1.00 | 1.00 | 1.00 | 7 |
| **Iris Versicolor** | 1.00 | 0.88 | 0.93 | 8 |
| **Iris Virginica** | 0.89 | 1.00 | 0.94 | 8 |
| **Macro Average** | **0.96** | **0.96** | **0.96** | **23** |
| **Weighted Average** | **0.96** | **0.96** | **0.96** | **23** |

---

## 9. Interactive Prediction

The web application provides real-time client-side forward-pass prediction:
- Sliders for Sepal Length, Sepal Width, Petal Length, and Petal Width.
- Preset buttons for benchmark samples:
  - **Setosa Sample:** `[5.0, 3.5, 1.4, 0.2]` $\to$ Predicts *Iris Setosa* ($99.99\%$ confidence).
  - **Versicolor Sample:** `[6.0, 2.9, 4.5, 1.3]` $\to$ Predicts *Iris Versicolor* ($99.99\%$ confidence).
  - **Virginica Sample:** `[6.7, 3.1, 5.6, 2.4]` $\to$ Predicts *Iris Virginica* ($99.99\%$ confidence).
- Visual horizontal Softmax probability bars summing to exactly $100\%$.
- Technical vector readout displaying the raw vector $x$ and the standardized vector $z$.

---

## 10. Technology Stack

### Production Frontend (Vercel Deployed)
- **Framework:** Next.js 15 (App Router)
- **Library:** React 19
- **Language:** TypeScript 5
- **Styling:** Tailwind CSS 3.4
- **Icons:** Lucide React
- **Hosting / CDN:** Vercel Edge Network

### Development & Offline Training
- **Deep Learning Framework:** TensorFlow 2.x / Keras
- **Machine Learning & Preprocessing:** Scikit-learn
- **Data Analysis:** Pandas, NumPy

---

## 11. Project Structure

```
.
├── app/
│   ├── globals.css              # Custom Tailwind styling & design tokens
│   ├── layout.tsx               # Root application layout with metadata
│   └── page.tsx                 # Main dashboard integrating all 7 tabs
├── components/
│   ├── ArchitectureDiagram.tsx  # Interactive SVG neural network topology & 515 param table
│   ├── ConfusionMatrix.tsx      # 3x3 Heatmap for holdout test evaluation
│   ├── FeatureDistributions.tsx # Feature distribution & range charts
│   ├── MetricCard.tsx           # Reusable metric card with highlight states
│   ├── Navbar.tsx               # Navigation header with tabs & student USN badge
│   ├── Predictor.tsx            # Real-time inference sliders, presets & Softmax bars
│   └── TrainingCurves.tsx       # SVG training & validation loss/accuracy curves
├── lib/
│   └── inference.ts             # Exact TypeScript forward-pass inference engine
├── public/
│   ├── data/
│   │   ├── dataset.json         # 150 Iris records for table & distribution analysis
│   │   ├── evaluation.json      # Holdout test evaluation, CM, classification report
│   │   ├── preprocessing.json   # Split counts, scaler mean/scale, sample z-scores
│   │   └── training_history.json# Actual epoch-by-epoch loss & accuracy metrics
│   └── model/
│       └── model.json           # 515 weights & biases, scaler params, class metadata
├── training/
│   └── train_model.py           # Standalone Python script for training & artifact export
├── legacy_streamlit/            # Archived original Streamlit prototype files
│   ├── app.py
│   └── .streamlit/
├── model_utils.py               # Original Python utility module
├── visualizations.py            # Original Python visualization module
├── next.config.mjs              # Next.js configuration
├── package.json                 # Next.js npm dependencies and scripts
├── postcss.config.mjs           # PostCSS Tailwind configuration
├── tailwind.config.ts           # Tailwind theme extensions
├── tsconfig.json                # Strict TypeScript configuration
├── requirements.txt             # Python offline training dependencies
├── .gitignore                   # Excludes node_modules, .next, and python cache
└── README.md                    # Comprehensive documentation
```

---

## 12. Local Development

### Prerequisites
- Node.js 18+ (tested with v20 and v24)
- npm 9+
- Python 3.10+ (only required if retraining the model offline)

### Step 1: Clone Repository & Install Node Dependencies
```bash
git clone https://github.com/AmoghSamarth/Deep-feedforward-neural-network-for-classification.git
cd Deep-feedforward-neural-network-for-classification
npm install
```

### Step 2: (Optional) Re-train the Model Offline
If you wish to re-train the Keras model with fresh weights and regenerate all JSON artifacts:
```bash
pip install -r requirements.txt
python training/train_model.py
```

### Step 3: Run the Next.js Development Server
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser to view the application.

### Step 4: Build for Production
```bash
npm run build
npm run start
```

---

## 13. Production Deployment

The project is built as a static/hybrid Next.js application. Production builds execute:
```bash
npm run build
```
This generates an optimized production bundle with pre-rendered pages, sub-130 KB First Load JavaScript, and zero serverless runtime requirements.

---

## 14. Vercel Deployment

Deploying this repository to **Vercel** is instantaneous:

1. Push your changes to GitHub:
   ```bash
   git add .
   git commit -m "Deploy Next.js FNN classification app to Vercel"
   git push origin main
   ```
2. Log into your [Vercel Dashboard](https://vercel.com).
3. Click **Add New Project** and select this GitHub repository.
4. Vercel automatically detects the **Next.js** framework preset:
   - **Framework Preset:** Next.js
   - **Root Directory:** `./`
   - **Build Command:** `next build` (default)
   - **Output Directory:** `.next` (default)
   - **Install Command:** `npm install` (default)
5. Click **Deploy**.

The site will build in under 1 minute and provide an instant production URL with SSL.

---

## 15. Model Inference Architecture

In production on Vercel, client-side inference runs via `lib/inference.ts`:

$$\text{Raw Features } [x_1, x_2, x_3, x_4] \xrightarrow{\text{StandardScaler}} [z_1, z_2, z_3, z_4] \xrightarrow{\text{FNN Forward Pass}} \text{Logits} \xrightarrow{\text{Softmax}} \text{Probabilities}$$

1. **Zero Cold-Starts:** No Python virtual environments or serverless functions need to initialize.
2. **Deterministic Parity:** Uses the exact 515 floating-point weights and biases ($W^{[1..4]}, b^{[1..4]}$) and StandardScaler parameters ($\mu, \sigma$) exported from the TensorFlow/Keras training run.
3. **Numerically Stable Softmax:** Computed via $\sigma(z)_i = \frac{e^{z_i - \max(z)}}{\sum e^{z_j - \max(z)}}$ to eliminate floating-point overflow.
4. **Instant Response:** Forward propagation completes in under $0.5$ milliseconds directly in the client's browser.

---

## 16. Viva Questions & Answers (Quick Reference)

### Q1: Why is it called a "Feedforward" Neural Network?
**Answer:** In an FNN, information moves strictly forward from the input layer through hidden layers to the output layer ($x \to h_1 \to h_2 \to h_3 \to \hat{y}$). There are no feedback loops, cyclical paths, or recurrent connections.

### Q2: Why is Softmax used in the output layer instead of Sigmoid?
**Answer:** Sigmoid treats each output independently in $[0, 1]$, which is appropriate for binary or multi-label classification. Softmax is a generalization of Sigmoid for multi-class classification ($K \ge 3$) that normalizes outputs so that $\sum_{i=1}^K \sigma(z)_i = 1.0$, guaranteeing mutually exclusive class probabilities.

### Q3: Why is Categorical Cross-Entropy preferred over Mean Squared Error (MSE)?
**Answer:** Cross-Entropy is derived from maximum likelihood estimation under a multinomial distribution. When paired with Softmax, its partial derivative with respect to the logits is linear: $\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i$. MSE with Softmax causes vanishing gradients because its derivative involves $\hat{y}_i(1 - \hat{y}_i)$, which approaches zero when predictions are confident yet incorrect.

### Q4: Why must `StandardScaler` be fitted ONLY on the training data?
**Answer:** Fitting a scaler on the entire dataset causes **data leakage** because information about the mean ($\mu$) and standard deviation ($\sigma$) of the unseen test set bleeds into the training process, leading to overly optimistic performance estimates. To ensure scientific rigor, the scaler is fitted strictly on the 104 training instances and then applied to validation and test instances.

### Q5: What is the purpose of EarlyStopping?
**Answer:** EarlyStopping monitors validation loss across epochs. When validation loss ceases to decrease for a designated number of epochs (`patience=15`), training terminates and the model weights from the epoch with the lowest validation loss are restored, preventing the network from overfitting to training noise.

### Q6: How are the 515 trainable parameters calculated?
**Answer:** 
- Hidden Layer 1: $(4 \text{ inputs} \times 16 \text{ neurons}) + 16 \text{ biases} = 64 + 16 = \mathbf{80}$
- Hidden Layer 2: $(16 \text{ inputs} \times 16 \text{ neurons}) + 16 \text{ biases} = 256 + 16 = \mathbf{272}$
- Hidden Layer 3: $(16 \text{ inputs} \times 8 \text{ neurons}) + 8 \text{ biases} = 128 + 8 = \mathbf{136}$
- Output Layer: $(8 \text{ inputs} \times 3 \text{ neurons}) + 3 \text{ biases} = 24 + 3 = \mathbf{27}$
- **Total:** $80 + 272 + 136 + 27 = \mathbf{515 \text{ trainable parameters}}$.

### Q7: Why did we migrate from Streamlit to Next.js for Vercel deployment?
**Answer:** Streamlit is designed around a stateful Python server that continually reruns scripts on interaction, which is incompatible with Vercel's stateless serverless and static CDN architecture. Migrating to Next.js decouples offline TensorFlow model training from client-side forward propagation, delivering sub-millisecond predictions, zero server costs, and seamless deployment on Vercel.

---

## 17. Author Details
- **Student Name:** Amogh Samarth
- **USN:** CM23034
- **Course:** Pattern Recognition
- **Institution Assignment:** TAE 1: Project Based Learning – Phase I
- **GitHub Repository:** [https://github.com/AmoghSamarth/Deep-feedforward-neural-network-for-classification](https://github.com/AmoghSamarth/Deep-feedforward-neural-network-for-classification)
