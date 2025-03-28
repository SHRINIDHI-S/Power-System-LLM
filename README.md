# Power-System-LLM
# 🔌 Hybrid Power System Failure Detection using LLM + ML

This project explores a hybrid modeling approach that combines domain knowledge from power systems literature (via LLM embeddings) with machine learning on numerical measurement data to detect failure events.

---

## 🚀 Project Overview

The goal is to:
- Simulate realistic power system measurement data (e.g., voltage, current, frequency)
- Use a lightweight open-source LLM to extract embeddings from textual domain knowledge
- Feed this combined information into a machine learning model to detect failure conditions

This mimics how an engineer would diagnose an issue: by **looking at data** and **recalling knowledge** from manuals, standards, and past experience.

---

## 📊 Components

### 1. Synthetic Power System Dataset
- Simulated using NumPy
- Features include:
  - `voltage_kV`, `current_A`, `frequency_Hz`, `power_MW`
  - Binary `failure` label (1 = failure, 0 = normal)

### 2. Domain Text Corpus
- Small set of example sentences from power systems literature
- Embedded using SentenceTransformers (`all-MiniLM-L6-v2`)

### 3. Machine Learning Model
- XGBoost classifier (or RandomForest as an alternative)
- Trained on numerical + embedding-augmented features
- Evaluated using accuracy, precision, recall, F1-score


---

## 🛠️ How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost sentence-transformers
