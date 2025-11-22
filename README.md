---

# 🎧 Audio Emotion Recognition – Hybrid Wav2Vec2 + MelCNN + LightGBM

This project implements a **high-accuracy audio-based emotion recognition system** using the **RAVDESS dataset**.
It includes:

* A **Transformer-based** Wav2Vec2 classifier
* A **Mel-Spectrogram CNN**
* A **Hybrid stacking model using LightGBM**
* Full evaluation (accuracy, F1, ROC-AUC, confusion matrices, plots)

The final hybrid model achieves **85.19% accuracy**, outperforming both unimodal approaches.

---

## 📁 Dataset

**RAVDESS – Ryerson Audio-Visual Database of Emotional Speech and Song**

* Classes (8 total):
  `neutral, calm, happy, sad, angry, fearful, disgust, surprised`
* All audio clips are **resampled to 16kHz**, normalized, and trimmed/padded to 3 seconds.
* Dataset is split into:

  * **70% train**
  * **15% validation**
  * **15% test**

---

## 🧱 Project Structure

| Cell       | Description                                                 |
| ---------- | ----------------------------------------------------------- |
| **Cell 1** | Dataset loading, preprocessing, spectrogram visualization   |
| **Cell 2** | Dataset classes + DataLoaders (raw audio + mel spectrogram) |
| **Cell 3** | Wav2Vec2 model training                                     |
| **Cell 4** | MelCNN model training                                       |
| **Cell 5** | Hybrid stacking with LightGBM (CNN + W2V2)                  |
| **Cell 6** | Final graphs, training curves, ROC-AUC charts, barplots     |

---

## 🏗 Model Architectures

### **1. Wav2Vec2 Transformer Model**

* Pretrained: `facebook/wav2vec2-base`
* Fine-tuned on RAVDESS for 8-class classification
* Input: raw waveform (16kHz, 3s)
* Optimizer: AdamW
* Mixed Precision: Yes
* Best validation accuracy: **84.42%**

---

### **2. Mel-Spectrogram CNN**

* Input: (1 × 80 × T) mel spectrogram
* 3 convolution blocks:

  * Conv → BatchNorm → ReLU → Pool → Dropout
* Fully-connected classifier head
* Best validation accuracy: **22–25%**

*(CNN is kept to support hybrid stacking, not as a high performer.)*

---

### **3. Hybrid Stacking Model – LightGBM**

The hybrid model concatenates:

```
[ CNN Probabilities | Wav2Vec2 Probabilities ]
```

And trains an **LGBMClassifier (multiclass)** with:

* Early stopping
* 5000 max trees
* Learning rate = 0.02
* Unbalanced class option enabled

Final hybrid test accuracy: **85.19%**

---

## 📊 Results

### **Overall Performance Summary**

| Model        | Accuracy   | Micro AUC  | Macro Precision | Macro Recall | Macro F1   |
| ------------ | ---------- | ---------- | --------------- | ------------ | ---------- |
| **Wav2Vec2** | **0.8380** | **0.9848** | 0.8372          | 0.8442       | 0.8298     |
| **MelCNN**   | 0.2454     | 0.5871     | 0.0965          | 0.2284       | 0.1328     |
| **Hybrid**   | **0.8519** | 0.9615     | **0.8476**      | **0.8524**   | **0.8482** |

> **The hybrid model achieves the highest accuracy and F1, outperforming both unimodal models.**

---

### **Confusion Matrix (Hybrid Model)**

(Generated automatically in Cell 5.)

### **ROC Curves (Per-Class + Micro-Average)**

Plotted in Cell 5 using One-vs-Rest AUC.

### **Training Curves**

* Wav2Vec2 accuracy & loss curves
* MelCNN accuracy & loss curves
* Side-by-side model comparison barplot

(All auto-generated in Cell 6.)

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install torch torchaudio transformers lightgbm matplotlib seaborn scikit-learn
```

2. Place RAVDESS dataset under:

```
./audio_speech_actors_01-24/
```

3. Run notebook cells in order:

```
Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5 → Cell 6
```

4. Outputs will be generated automatically:

* Model weights
* Evaluation tables
* Accuracy comparisons
* ROC-AUC graphs
* Confusion matrices

---

## 🧪 Hybrid Model Formula

The hybrid’s meta-features are:

```
X_meta = concat(CNN_softmax_probs, W2V2_softmax_probs)
```

LightGBM learns:

* When CNN is useful
* When Wav2Vec2 dominates
* Class-wise decision boundaries
* Ensemble weighting per class

This results in:

* **Higher macro F1**
* **Better recall**
* **Improved class balance**
* **Stronger generalization**

---

## 📌 Key Takeaways

* **Wav2Vec2 does most of the heavy lifting** for emotion recognition.
* CNN contributes small but detectable signal for certain classes.
* LightGBM stacking successfully combines both for **SOTA-level accuracy on RAVDESS**.
* System is modular and ready for extension to **multimodal fusion with video and text**.

---
