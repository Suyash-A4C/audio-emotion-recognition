# Audio-Based Emotion Classification using RAVDESS  
Project 2: Audio Cyberbullying Detection (Audio Emotion Recognition Baseline)

## Overview
This project builds a complete audio emotion classification pipeline using the RAVDESS dataset.  
It serves as the audio component for a larger multimodal cyberbullying detection system.

The project includes:
1. Audio loading and preprocessing  
2. Feature extraction (Mel Spectrograms and MFCCs)  
3. Model training using two architectures:  
   - Convolutional Neural Network (AudioCNN)  
   - Recurrent Neural Network with LSTM (AudioLSTM)  
4. Hybrid stacking-based classifier using LightGBM  
5. Model evaluation and comparisons (Accuracy, F1-scores, ROC AUC)

---

## Dataset
**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**  
- 1440 audio files  
- 24 actors (12 male, 12 female)  
- 8 emotion classes:
  - angry  
  - calm  
  - disgust  
  - fearful  
  - happy  
  - neutral  
  - sad  
  - surprised  

Each filename encodes:
`modality - channel - emotion - intensity - statement - repetition - actor`

Example:  
`03-01-06-01-02-01-12.wav`

Dataset split:
- Train: ~70 percent  
- Validation: ~15 percent  
- Test: ~15 percent  

Stratified splitting maintains class balance.

---

## Audio Preprocessing

### Standardization Steps
- Resample all audio to **16 kHz**
- Convert to **mono**
- Normalize waveform amplitude  
- Pad or truncate to **3 seconds fixed duration**
- Convert into feature representations:
  - **Mel Spectrograms** (80 mel bins)
  - **MFCCs** (80 coefficients for RNN)

### Feature Dimensions
- Mel Spectrogram shape: `(1, 80, 301)`  
- MFCC (RNN input): `(time_frames, 80)`

---

## Models Implemented

### 1. Audio CNN (Mel Spectrogram Input)
Architecture:
- 3 convolutional blocks:
  - Conv2D → BatchNorm → ReLU → MaxPool  
- Dropout = 0.3  
- Fully connected layers:
  - FC(Flattened → 256)  
  - FC(256 → 8 output classes)

Training:
- Epochs: 20  
- Optimizer: Adam  
- Loss: CrossEntropy  
- Scheduler: ReduceLROnPlateau  

Final Test Accuracy: **48.15 percent**  
Micro-Average ROC AUC: **0.8597**

---

### 2. Audio LSTM (MFCC Input)
Architecture:
- 2-layer bidirectional LSTM  
- Hidden size: 128  
- Dropout: 0.4  
- Final FC layer → 8 classes  

Training:
- Epochs: 20  
- Optimizer: Adam  
- Loss: CrossEntropy  
- Scheduler: ReduceLROnPlateau  

Final Test Accuracy: **41.20 percent**  
Micro-Average ROC AUC: **0.8203**

---

### 3. Hybrid Ensemble (CNN + LSTM stacked with LightGBM)
Approach:
- Extract softmax probability outputs from:
  - Trained CNN  
  - Trained LSTM  
- Concatenate into 16-dimensional feature vector per sample  
- Train LightGBM meta-classifier with:
  - 2000 estimators  
  - Learning rate 0.01  
  - is_unbalance=True  

Final Test Accuracy: **51.85 percent**  
Micro-Average ROC AUC: **0.8103**

---

## Results Summary

| Model                                   | Accuracy | Micro-AUC |
|----------------------------------------|----------|-----------|
| AudioCNN                               | 0.4815   | 0.8597    |
| AudioLSTM                              | 0.4120   | 0.8203    |
| Hybrid CNN + LSTM + LightGBM (AUC tuned) | 0.5185 | 0.8103    |

Key observations:
- CNN outperforms LSTM on raw accuracy and AUC.  
- LSTM performs well on calm and disgust but struggles on happy and neutral.  
- Hybrid LightGBM improves overall accuracy to 51.85 percent.  
- CNN’s Mel Spectrogram features appear more robust for this dataset.

---

## Visualizations
The notebook includes:
- Waveform plots
- Mel Spectrograms
- Confusion matrices for each model
- ROC curves (One-vs-Rest and Micro-average)
- Comparative bar charts for accuracy and AUC

---

## How to Run

1. Install dependencies:
pip install torch torchaudio librosa numpy pandas matplotlib seaborn scikit-learn lightgbm

2. Place dataset directory in your project folder:

3. Run notebook sequentially:
- Cell 1: Load dataset + preprocessing  
- Cell 2: Dataset class + CNN model + Training  
- Cell 3: CNN evaluation  
- Cell 4: LSTM model + Training + Evaluation  
- Cell 5: Hybrid model (LightGBM)  
- Cell 6: Final comparisons  

---

## Folder Structure
project/
│── audiomodel.ipynb
│── audio_speech_actors_01-24/
│── README.md

---

## Conclusion
This project provides a full audio emotion classification pipeline using deep learning and ensemble methods.

The Hybrid CNN + LSTM + LightGBM model achieves the best accuracy (51.85 percent), forming a strong baseline for integrating audio into a multimodal cyberbullying detection system.

---

## Future Improvements
- Use pretrained audio embedding models:  
  - YAMNet  
  - VGGish  
  - Wav2Vec2  
- Add SpecAugment for data augmentation  
- Fine-tune architectures for emotion classification  
- Extend to multimodal fusion (text + audio + video)


