# Audio-Based Emotion Classification using RAVDESS  
Project 2: Audio Cyberbullying Detection (Audio Emotion Recognition Baseline)

## Overview
This project builds a complete **audio emotion classification pipeline** using the RAVDESS dataset.  
It serves as the **audio component** for a larger **multimodal cyberbullying detection system**.

The project includes:

1. Audio loading and preprocessing  
2. Feature extraction (Mel Spectrograms and MFCCs)  
3. Model training using three architectures:  
   - **AudioCNN** (Convolutional Neural Network on Mel Spectrograms)  
   - **AudioLSTM** (Bidirectional LSTM on MFCC sequences)  
   - **Hybrid CNN–LSTM + LightGBM** (deep feature extractor + gradient boosting)  
4. Model evaluation and comparisons (Accuracy, F1-scores, ROC curves)  
5. Visualization of training dynamics and confusion matrices

---

## Dataset

**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**  

- **1440** audio files  
- **24** actors (12 male, 12 female)  
- **8 emotion classes:**
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

**Dataset split:**

- Train: ~70%  
- Validation: ~15%  
- Test: ~15%  

Stratified splitting is used to **maintain class balance** across splits.

---

## Audio Preprocessing

All audio preprocessing and feature extraction is implemented using **PyTorch** and **torchaudio**, with some visualization support from **librosa**.

### Standardization Steps

For every audio file:

- Resample to **16 kHz** (`TARGET_SAMPLE_RATE = 16000`)  
- Convert to **mono** (average across channels if stereo)  
- Normalize waveform amplitude to [-1, 1]  
- **Pad or truncate to 3 seconds** fixed duration  
  - Target number of samples: `AUDIO_DURATION * TARGET_SAMPLE_RATE`  
- Convert into feature representations:
  - **Mel Spectrograms** with:
    - `n_mels = 80`
    - `n_fft = 400`
    - `hop_length = 160`  
  - **MFCCs** with:
    - 80 MFCC coefficients
    - Same FFT, hop length, and mel parameters

### Feature Dimensions

- **Mel Spectrogram (for CNN models)**  
  - Shape per sample: `(1, 80, ~301)`  
  - 1 channel, 80 mel frequency bins, ~301 time frames for a 3-second clip.

- **MFCCs (for LSTM models)**  
  - Shape per sample: `(time_frames ≈ 301, 80)`  
  - Interpreted as a **sequence of 301 time steps**, each with 80-dimensional MFCC features.

A custom `RAVDESS_AudioDataset` class handles:

- Loading audio from disk  
- Resampling, mono conversion, normalization  
- Padding/truncation to fixed length  
- On-the-fly Mel Spectrogram or MFCC computation  
- Returning `(features, label)` pairs ready for model consumption.

Class weights are also computed from label frequencies and passed to `CrossEntropyLoss` to handle **class imbalance**.

---

## Models Implemented

### 1. AudioCNN (Mel Spectrogram Input)

**Input:** Mel Spectrograms with shape `(batch_size, 1, 80, time_frames)`  

**Architecture:**

- **4 convolutional blocks:**
  1. `Conv2d(1 → 32, kernel_size=3x3, padding=1)`  
     → BatchNorm2d → ReLU → MaxPool2d(2x2) → Dropout(0.3)  
  2. `Conv2d(32 → 64, 3x3, padding=1)`  
     → BatchNorm2d → ReLU → MaxPool2d(2x2) → Dropout(0.3)  
  3. `Conv2d(64 → 128, 3x3, padding=1)`  
     → BatchNorm2d → ReLU → MaxPool2d(2x2) → Dropout(0.3)  
  4. `Conv2d(128 → 256, 3x3, padding=1)`  
     → BatchNorm2d → ReLU → MaxPool2d(2x2) → Dropout(0.3)

- **Fully Connected Head:**
  - Flatten  
  - `Linear(flattened_features → 512)`  
    → BatchNorm1d → ReLU → Dropout(0.3)  
  - `Linear(512 → 8)` (one logit per emotion class)

The flattened feature size is computed **dynamically** using a dummy input, so the model adapts automatically if the time dimension changes.

**Training Setup:**

- Loss: `CrossEntropyLoss` with **class weights**  
- Optimizer: `Adam` (`lr = 0.001`)  
- Epochs: **20**  
- Batch size: **32**  
- Device: GPU (`cuda`) if available, else CPU  
- Best model weights chosen based on **validation accuracy**

**Final Test Performance:**

- **Test Accuracy:** **0.5602** (56.02%)

The CNN remains the best-performing model in terms of raw test accuracy, effectively exploiting the spatial structure of Mel Spectrograms.

---

### 2. AudioLSTM (MFCC Sequence Input)

**Input:** MFCC sequences with shape  
`(batch_size, seq_len ≈ 301, input_dim = 80)`

**Architecture:**

- **Bidirectional LSTM:**
  - Input size: `80` (MFCC features per time step)  
  - Hidden size: `128`  
  - Number of layers: `2`  
  - Dropout: `0.4` (applied between LSTM layers)  
  - Bidirectional: **True** (forward + backward)

- **Classification Head:**
  - Use the **last time step output** from the final Bidirectional LSTM layer  
    → Shape: `(batch_size, hidden_size * num_directions)` = `(batch_size, 256)`  
  - Dropout(0.4)  
  - `Linear(256 → 8)` for class logits  

**Training Setup:**

- Loss: `CrossEntropyLoss` with **class weights**  
- Optimizer: `Adam` (`lr = 0.001`)  
- Epochs: **20**  
- Batch size: **32**  
- Best model based on validation accuracy

**Final Test Performance:**

- **Test Accuracy:** **0.4907** (49.07%)

The LSTM captures **temporal dynamics** in the MFCC sequence and performs competitively, but still lags behind the CNN in overall accuracy.

---

### 3. Hybrid Model: CNN–LSTM Feature Extractor + LightGBM

This is a **two-stage hybrid architecture**:

1. A deep **CNN–LSTM feature extractor** in PyTorch  
2. A **LightGBM classifier** trained on the extracted deep features

#### 3.1 CNN–LSTM Feature Extractor

**Input:** Mel Spectrograms `(batch_size, 1, 80, time_frames)`

**CNN Feature Extractor:**

- Conv2d(1 → 32, 5x5, padding=2) → BatchNorm2d → ReLU → MaxPool2d(2x2) → Dropout(0.3)  
- Conv2d(32 → 64, 3x3, padding=1) → BatchNorm2d → ReLU → MaxPool2d(2x2) → Dropout(0.3)  
- Conv2d(64 → 128, 3x3, padding=1) → BatchNorm2d → ReLU → MaxPool2d(2x2) → Dropout(0.3)

The CNN outputs a feature map of shape:
`(batch_size, channels=128, height, width)`

This is then **reshaped into a sequence** for the LSTM:

- Sequence length: `width` (time dimension after pooling)  
- Features per step: `channels * height`

**LSTM over CNN Features:**

- Input size: `lstm_input_features = channels * height`  
- Hidden size: `128`  
- Layers: `2`  
- Bidirectional: **True**  
- Dropout: `0.4`

The **final hidden states** from all layers and directions are concatenated:

- Shape: `(batch_size, 512)`  

This 512-dimensional vector is used as the **deep feature representation** for each audio clip.

To train this feature extractor, a simple linear classifier is attached:

- `Linear(512 → 8)` (for internal training)  

The combined model is trained end-to-end (CNN + LSTM + linear head) using:

- Loss: `CrossEntropyLoss` with class weights  
- Optimizer: `Adam` (`lr = 0.0003`)  
- Epochs: **35**

After training, the linear classifier is discarded and the CNN–LSTM feature extractor is used to **extract 512-D features** for all train/val/test samples.

#### 3.2 LightGBM Classifier

Using the extracted features:

- `X_train`: shape `(1008, 512)`  
- `X_val`: shape `(216, 512)`  
- `X_test`: shape `(216, 512)`

A **LightGBM multiclass classifier** is trained with:

- Objective: `multiclass`  
- `num_class = 8`  
- `n_estimators = 500`  
- `learning_rate = 0.05`  
- `num_leaves = 31`  
- `colsample_bytree = 0.7`  
- `subsample = 0.7`  
- `reg_alpha = 0.1`  
- `reg_lambda = 0.1`  
- `early_stopping_round` on a validation set  

**Final Test Performance:**

- **Test Accuracy:** **0.5509** (55.09%)

The hybrid model is **very close to the CNN** in accuracy, but benefits from:

- Deep learned representations (CNN–LSTM)  
- Flexible decision boundaries from LightGBM  

This makes it a strong, robust baseline for the audio branch.

---

## Results Summary

### Final Test Accuracies

| Model                                      | Test Accuracy |
|-------------------------------------------|---------------|
| **AudioCNN (Mel Spectrogram)**            | **0.5602**    |
| **Hybrid CNN–LSTM + LightGBM (Features)** | **0.5509**    |
| **AudioLSTM (MFCC Sequence)**             | **0.4907**    |

### Key Observations

- The **AudioCNN** achieves the **highest overall accuracy (~56%)**, confirming that 2D convolution over Mel Spectrograms is very effective for RAVDESS emotion recognition.
- The **Hybrid CNN–LSTM + LightGBM** is competitive with the CNN and slightly behind it in raw accuracy, but benefits from:
  - Rich temporal–spectral features from the CNN–LSTM extractor  
  - Strong non-linear decision boundaries from LightGBM.
- The **AudioLSTM** model, while lower in accuracy (~49%), still provides valuable insight into the sequential MFCC representation and class-specific behavior.
- Class-weighted loss helps address class imbalance across all models.

---

## Visualizations

The notebook includes multiple visualizations:

### Data and Feature Exploration

- **Sample Waveform + Mel Spectrogram** for a test audio file  
- **Emotion distribution by gender** (male vs female actors)  
- **Mel Spectrograms for different emotions** (e.g., happy, sad, angry, neutral)  
- **Overall gender distribution** in the dataset  

### Model Evaluation Plots

For each model (CNN, LSTM, Hybrid):

- Confusion matrices  
- Per-class ROC curves  

For comparison:

- Training vs validation **loss curves** across all three models  
- Training vs validation **accuracy curves** across all three models  
- **Bar plot** comparing final test accuracies  
- **Side-by-side confusion matrices** (CNN, LSTM, Hybrid)  
- **Macro-average ROC curves** for all models on a single plot

---

## How to Run

### 1. Install Dependencies

Make sure you have Python 3.x with CUDA-enabled PyTorch if you want GPU training.

Install core dependencies:

```bash
pip install torch torchaudio librosa numpy pandas matplotlib seaborn scikit-learn lightgbm

2. Folder Setup

Place the RAVDESS audio folder next to your notebook/script:

project/
│── audiomodel.ipynb  (or .py script)
│── audio_speech_actors_01-24/
│── READMEnoprob.md

In the code, the dataset path is configured via:

RAVDESS_ROOT_DIR = "."

AUDIO_DATA_DIR = "audio_speech_actors_01-24"

3. Run the Notebook Sequentially

The code is organized into Consolidated Blocks:

Block 1 – Data preprocessing and exploration

Loads RAVDESS, parses metadata, computes class weights

Creates train/validation/test splits

Visualizes waveform, Mel Spectrograms, and dataset stats

Block 2 – Model 1: AudioCNN

Creates DataLoaders with Mel Spectrograms

Defines and trains the AudioCNN

Evaluates on test set and plots metrics

Block 3 – Model 2: AudioLSTM

Creates DataLoaders with MFCC sequences

Defines and trains the AudioLSTM

Evaluates on test set and plots metrics

Block 4 – Hybrid: CNN–LSTM Feature Extractor + LightGBM

Defines CNN–LSTM feature extractor

Trains full PyTorch model with internal classifier

Extracts deep features for all splits

Trains LightGBM and evaluates on test data

Block 5 – Comparison & Summary

Plots loss and accuracy curves for all models

Shows side-by-side confusion matrices

Plots macro-average ROC curves

Summarizes test accuracies in a table and bar chart

Run each block in order; all configuration is centralized in the AudioConfig class.
Conclusion

This project implements a complete audio emotion recognition pipeline on the RAVDESS dataset, using:

A strong AudioCNN baseline

A sequence-based AudioLSTM model

A hybrid CNN–LSTM + LightGBM architecture that combines deep learned features with gradient boosting.

The best-performing model in terms of test accuracy is the AudioCNN, achieving 56.02% accuracy, with the hybrid model close behind at 55.09%.
These results form a solid audio baseline for integration into a multimodal cyberbullying detection system (audio + text + video).

Future Improvements

Use pretrained audio embedding models:

YAMNet

VGGish

Wav2Vec2 or other self-supervised audio encoders

Add SpecAugment and other data augmentation strategies

Hyperparameter tuning for:

CNN/LSTM architectures

Learning rates, schedulers, and regularization

LightGBM parameters

Explore cross-validation for more robust estimates

Extend to multimodal fusion:

Combine this audio branch with text and video branches

Late fusion (logit-level) or joint representation learning