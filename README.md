# 💡 White Balance Regression Model (EfficientNetV2-S)
## 🎯 Project Overview
This project implements a deep learning solution for the White Balance Regression task, aiming to predict the precise Color Temperature (Kelvin) and Tint values required to correct image color casts. The core solution utilizes a fine-tuned EfficientNetV2-S model trained across a multi-stage process.

## 🔬 1. Technical Strategy and Model Architecture
### 1.1 Model Backbone and Adaptation
* Architecture: EfficientNetV2-S (pre-trained on ImageNet).
* Task: Regression, predicting two continuous parameters: Scaled Temperature and Scaled Tint.
* Adaptation: The model's final classification head was replaced with a Linear Regression Head with two outputs.
### 1.2 Multi-Stage Training Methodology
Training was strategically executed in two phases to maximize both stability and final precision:
* Phase 1: Generalization 256 x 256: Initial training at a lower resolution to quickly establish robust, generalized features and stable convergence principles.
* Phase 2: Fine-Tuning 384 x 384: Loading the best checkpoint from Phase 1 and resuming training at a higher resolution. A very low learning rate ($\mathbf{1e-5}$) was crucial here to ensure weight adjustments were minimal and precise, focusing on capturing fine-grained color details.
### 1.3 Key Optimization Decisions
|Component|Setting/Decision             |Impact/Rationale|
|---------|-----------------------------|----------------|
|Loss Function|Mean Squared Error (MSE)     |Chosen to heavily penalize large errors, particularly targeting and reducing high Temperature outliers (e.g., predictions far outside the mean Kelvin range).|
|Augmentations|ColorJitter                  |Essential for teaching the model color invariance. Randomly altering colors forces the model to learn the intrinsic scene illumination, ignoring spurious color noise.|
|Prediction Handling|Clipping and Rounding        |Final predictions are clipped to the valid range (1500 K to 10000 K) and rounded to the nearest integer to meet common formatting requirements.|

## 📈 2. Project Outcomes and Performance
The model's success is measured by the Mean Absolute Error (MAE) between the predicted and ground-truth parameters.
Final Performance

* Temp MAE: **575.46**
* Tint MAE: **6.03**
* Avg MAE:  **290.75**

## 📊 3. Performance Visualizations: Training and Fine-Tuning Graphs

### model Performance Graph for Training Dataset
<img width="1389" height="390" alt="model_tarin_graph" src="https://github.com/user-attachments/assets/f936fe2b-74e0-437f-b701-88196c977ad3" />

### Fine-Tuning model Performance Graph for Training Dataset
<img width="1389" height="390" alt="finetune_train_graph" src="https://github.com/user-attachments/assets/4ceaff0d-d753-4f95-84af-2a6cfb63309b" />


## Finding: Stability vs. Accuracy
A critical decision point involved managing the risk of overfitting during the 384 x 384 fine-tuning:
* **Risk**: Aggressive fine-tuning at high resolution with a low learning rate can lead to memorization of dataset noise.
* **Mitigation**: The robust 256 x 256 checkpoint (model_new_woo.pth) was maintained as a reliable fallback. The final submission model was selected based on the checkpoint demonstrating the best combination of low validation error and high stability.

## 💻 4. Hybrid Execution Environment
The project leveraged a hybrid hardware setup to successfully execute demanding training runs despite local limitations.
|Environment|GPU                          |Purpose|Optimization                                                                                    |
|-----------|-----------------------------|-------|------------------------------------------------------------------------------------------------|
|Local Development|NVIDIA MX330 (2 GB VRAM)     |Code development, custom data handler debugging, and stability verification.|Required using a conservative BATCH_SIZE=8 (or lower).                                          |
|Cloud Acceleration|NVIDIA Tesla T4 (or similar via Colab)|High-speed, multi-epoch training and fine-tuning.|Required setting NUM_WORKERS≥4 to prevent GPU starvation and eliminate data loading bottlenecks.|

# 🔗 Project Assets and Resources
|Resource|Link                         |
|--------|-----------------------------|
|GITHUB REPO|https://github.com/pratim4dasude/White_Balance|
|Google Colab Notebook|https://colab.research.google.com/drive/12sXPSsn9JcrgueQHW_Q1JD2RYpnGL0oA?usp=sharing|
|Full Dataset|https://www.kaggle.com/datasets/pratimdasude/white-balance-dataset|
|Trained Model Checkpoint|https://www.kaggle.com/models/pratimdasude/white-balance-models|
|Presentation |https://docs.google.com/presentation/d/1WOGKSqtsjmK3JTbsLaNnZ4QDZAyHV8ZbvDM5H73XmpU/edit?usp=sharing|


