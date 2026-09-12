# Deep Feedforward Neural Network for Classification

**Course:** Pattern Recognition  
**Assessment:** TAE 1: Project Based Learning – Phase I  
**Student Name:** Amogh Samarth  
**USN:** CM23034  

---

## 📖 Project Overview
This project presents a minimal, modern, and interactive demonstration of a **Deep Feedforward Neural Network (FNN)** for multi-class classification on the **Iris dataset**.

## 🧠 Neural Network Architecture
$$\\text{Input (4 Features)} \\to \\text{Dense (16, ReLU)} \\to \\text{Dense (16, ReLU)} \\to \\text{Dense (8, ReLU)} \\to \\text{Dense (3, Softmax)}$$

## ⚙️ Data Preprocessing Pipeline
1. **Dataset Loading:** 150 Iris flower samples.
2. **Missing Values Check:** 0 null values verified.
3. **StandardScaler Normalization:** Fitted strictly on training partition.
4. **Stratified Partition:** 70% Train, 15% Validation, 15% Test.
5. **Target Encoding:** One-Hot categorical vectors.

## 🚀 Installation & Execution
```bash
pip install -r requirements.txt
streamlit run app.py
```
