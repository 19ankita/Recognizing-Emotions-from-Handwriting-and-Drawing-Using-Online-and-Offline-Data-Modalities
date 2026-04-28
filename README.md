
📌 This project is based on my master’s thesis, which investigates whether human emotional states (stress, anxiety, depression) can be inferred from handwriting and drawing behavior.

The work leverages the EMOTHAW dataset, combining:

- 🟢 Online data → temporal signals (pressure, speed, stroke dynamics)
- 🔵 Offline data → static handwriting images

A key research question:

Can static handwriting images alone capture emotional information typically present in dynamic writing signals?

🎯 Core Contributions 

As defined in the thesis:

- 1. Regression on Online Data
Predict continuous DASS scores using handcrafted features
- 2. Transfer Learning on Offline Images
CNN-based classification + regression
Uses pretrained ResNet
- 3. Comparative Analysis
Online vs Offline modalities
Evaluates whether offline data alone is sufficient

📂 Dataset: EMOTHAW
- 7 handwriting & drawing tasks
- Collected using digital tablet
- Annotated with:
 - Depression
 - Anxiety
 - Stress (DASS scores)
Modalities:
- Online signals:
 - Pen coordinates
 - Pressure
 - Time
-Offline images:
 - Reconstructed grayscale handwriting
