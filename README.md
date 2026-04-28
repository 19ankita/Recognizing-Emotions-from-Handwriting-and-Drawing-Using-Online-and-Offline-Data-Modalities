## 📌 Thesis Motivation

This project is based on my master’s thesis, which investigates whether human emotional states — **stress, anxiety, and depression** — can be inferred from handwriting and drawing behavior.

The work leverages the **EMOTHAW dataset**, combining:

- 🟢 **Online data** → temporal signals (pressure, speed, stroke dynamics)  
- 🔵 **Offline data** → static handwriting images  

### ❓ Key Research Question

> Can static handwriting images alone capture emotional information typically present in dynamic writing signals?

---

## 🎯 Core Contributions

As defined in the thesis:

### 1. 📈 Regression on Online Data
- Predict continuous **DASS scores** using handcrafted features  
- Utilizes temporal handwriting dynamics (pressure, velocity, stroke patterns)

### 2. 🧠 Transfer Learning on Offline Images
- CNN-based **classification + regression**
- Uses pretrained **ResNet**
- Learns emotional patterns directly from static handwriting images  

### 3. ⚖️ Comparative Analysis
- Systematic comparison of **online vs offline modalities**
- Evaluates whether **offline data alone is sufficient** for emotion recognition  

---

## 📂 Dataset: EMOTHAW

The project uses the **EMOTHAW dataset**, designed for multimodal handwriting-based emotion analysis.

### ✍️ Data Collection
- 7 handwriting & drawing tasks  
- Collected using a **digital tablet**

### 🧾 Annotations
Each sample is labeled with:
- Depression  
- Anxiety  
- Stress  
(**DASS scores**)

### 🔍 Modalities

#### 🟢 Online Signals (Dynamic Data)
- Pen coordinates  
- Pressure  
- Time  

#### 🔵 Offline Images (Static Data)
- Reconstructed grayscale handwriting images

- ## ⚙️ Methodology (Thesis-Based)

---

## 🔹 1. Online Feature Engineering (Baseline + Regression)

The thesis utilizes **25 handcrafted features** to capture fine-grained behavioral dynamics in handwriting.

### 🧠 Feature Categories

- **Temporal** → writing time, stroke count  
- **Kinematic** → speed, acceleration  
- **Geometric** → writing angle, straightness  
- **Pressure** → mean pen pressure  
- **Spatial** → width, height, density  
- **Pen-up Dynamics** → off-surface motion  

👉 These features model **motor behavior**, which is closely linked to emotional states.

---

### 📈 Regression Models (Online Data)

The following models are implemented:

- Linear Regression  
- Ridge / Lasso / ElasticNet  
- Random Forest Regressor  
- Gradient Boosting  

### 🔄 Pipeline

1. **Standardization** (Z-score normalization)  
2. **PCA** (retain 95% variance)  
3. **Cross-validation + Hyperparameter tuning**  

### 📉 Loss Function
- Mean Squared Error (MSE)

### 📊 Evaluation Metrics
- RMSE (Root Mean Squared Error)  
- R² Score  

👉 Focus: Predicting **continuous emotional severity (DASS scores)** rather than only discrete labels.

---

## 🧠 2. Offline Image-Based Deep Learning

### 🔹 CNN + Transfer Learning

Due to limited dataset size, the thesis adopts a **transfer learning approach**:

### ✅ Strategy

- Pretrained **ResNet** (trained on ImageNet)  
- Backbone **fully frozen**  
- Replace final layer with **task-specific head**  

👉 This helps:
- Prevent overfitting  
- Leverage powerful pretrained visual representations  

---

## 🔹 Feature Fusion (Key Contribution)

A major contribution of the thesis is combining **visual** and **pseudo-dynamic** features.

### 🎨 CNN Visual Features
Extracted automatically:
- Edge structures  
- Stroke textures  
- Spatial writing patterns  

### 🔄 Pseudo-Dynamic Features (from Images)

Derived from static handwriting to approximate temporal behavior:

- Stroke length  
- Stroke thickness  
- Number of segments  
- Aspect ratio  
- Curvature  

👉 These features simulate **writing dynamics from static images**

---

