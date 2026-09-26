# 🎙️ Voice Cloning Detector

AI-powered synthetic voice detection system for identifying **AI-generated, cloned, and spoofed speech**.

This project is being developed as part of **Smart India Hackathon (SIH) 2026** for the problem of:

> **AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks**

The current implementation focuses on **Branch 1 — Synthetic Voice Detection**, using the pretrained **AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)** model.

---

## 🚀 Current Status

### Branch 1 — Synthetic Voice Detection

**Status: ✅ Working Prototype**

Current pipeline:

```text
Audio File
    ↓
Audio Preprocessing
    ↓
Mono Conversion
    ↓
Resampling to 16 kHz
    ↓
4.04-second Model Input
    ↓
Pretrained AASIST
    ↓
Softmax Classification
    ↓
Synthetic Probability
```

The current system runs locally on CPU and provides a synthetic/spoof probability for an input audio file.

---

## 🧠 Model

This project uses the pretrained:

**AASIST — Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks**

Paper:

> Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks", 2021.

AASIST is designed for detecting spoofed or synthetic speech in audio anti-spoofing scenarios.

### Model used

```text
Model: AASIST
Checkpoint: AASIST.pth
Sample Rate: 16,000 Hz
Input Samples: 64,600
Input Duration: ≈ 4.04 seconds
Device: CPU
```

The pretrained checkpoint is loaded locally from:

```text
aasist/models/weights/AASIST.pth
```

---

# 📁 Project Structure

```text
VOICE_CLONING_DETECTOR/
│
├── aasist/
│   ├── config/
│   │   └── AASIST.conf
│   │
│   ├── models/
│   │   └── weights/
│   │       ├── AASIST.pth
│   │       └── AASIST-L.pth
│   │
│   ├── models/
│   ├── data_utils.py
│   ├── main.py
│   ├── README.md
│   ├── LICENSE
│   └── NOTICE
│
├── synthetic_detection.py
├── test_synthetic.py
├── benchmark.py
├── test_windows.py
├── window_aggregation.py
├── create_stereo.py
├── developer_can_use.py
│
├── AASIST_NOTES.txt
├── Voice cloning detection documentation.pdf
├── .gitignore
└── README.md
```

The **root README** describes this project.

The original AASIST documentation remains inside:

```text
aasist/README.md
```

---

# ⚙️ Environment

The current implementation was developed and tested using:

```text
Python: 3.10.21
PyTorch: 2.11.0+cpu
Torchaudio: 2.11.0+cpu
Device: CPU
CUDA: Not available
```

The system can therefore be tested without an NVIDIA GPU.

---

# 🔧 Installation

## 1. Clone the repository

```bash
git clone https://github.com/AnuragAI2909/voice-cloning-detector.git
cd voice-cloning-detector
```

---

## 2. Create the Python environment

Using Conda:

```bash
conda create -n voice_clone python=3.10
conda activate voice_clone
```

---

## 3. Install dependencies

Install the required packages:

```bash
pip install torch torchaudio soundfile numpy
```

> The original AASIST repository contains its own dependency configuration. The current integration uses the packages required by the synthetic voice detection module.

---

# 🎧 Synthetic Voice Detection

The main integration function is:

```python
analyze_synthetic_voice(audio_file)
```

Example:

```python
from synthetic_detection import analyze_synthetic_voice

result = analyze_synthetic_voice("audio.wav")

print(result)
```

Example output:

```python
{
    "synthetic_probability": 0.9997,
    "bonafide_probability": 0.0003,
    "inference_time_seconds": 0.3877,
    "result": "SYNTHETIC / SPOOF"
}
```

---

# 🔍 How Detection Works

The system performs the following preprocessing steps.

### 1. Load audio

The input audio is loaded using `soundfile`.

### 2. Convert stereo to mono

If the input contains multiple channels, the channels are averaged into a single mono signal.

### 3. Resample

Audio is converted to:

```text
16,000 Hz
```

when the original sample rate is different.

### 4. Prepare model input

AASIST expects:

```text
64,600 samples
```

At 16 kHz this corresponds to approximately:

```text
64600 / 16000 ≈ 4.04 seconds
```

For longer audio, the current implementation uses the first 64,600 samples.

For shorter audio, the audio is repeated until the required input length is reached.

### 5. AASIST inference

The processed audio is passed through the pretrained AASIST model.

The model produces two output classes:

```text
Class 0 → Spoof
Class 1 → Bonafide
```

Softmax is then applied to obtain the class scores.

---

# 📊 Output

The detector returns:

| Field                    | Description                            |
| ------------------------ | -------------------------------------- |
| `synthetic_probability`  | Model-derived spoof/synthetic score    |
| `bonafide_probability`   | Model-derived real/bonafide score      |
| `inference_time_seconds` | Model inference time                   |
| `result`                 | Current threshold-based classification |

Example:

```json
{
    "synthetic_probability": 0.7552,
    "bonafide_probability": 0.2448,
    "inference_time_seconds": 0.4481,
    "result": "SYNTHETIC / SPOOF"
}
```

---

# 🎯 Initial Decision Threshold

The current prototype uses:

```text
synthetic_probability >= 0.5
        ↓
SYNTHETIC / SPOOF

synthetic_probability < 0.5
        ↓
BONAFIDE / REAL
```

### Important

The `0.5` threshold is currently a **prototype threshold**.

It has **not yet been formally calibrated or optimized** using a large validation dataset.

Therefore, this threshold should not be considered a production security threshold.

---

# ⚡ Performance

Initial local CPU testing showed approximately:

```text
Model inference:
~0.3–0.6 seconds per inference
```

A benchmark consisting of multiple warm-up and inference runs produced an average model inference time of approximately:

```text
0.31 seconds
```

Actual end-to-end latency will depend on:

* Audio decoding
* Resampling
* Hardware
* Operating system
* Python overhead
* Input handling
* Future preprocessing/VAD
* Future API/network communication

The benchmark represents **model inference performance**, not complete production latency.

---

# 🧪 Testing

Several types of audio have been tested during development, including:

* Real speech
* Synthetic speech
* Short audio
* Long audio
* 48 kHz audio
* Stereo audio
* Compressed/WhatsApp audio

Example observed outputs included:

```text
Real audio
synthetic_probability ≈ 0.0001

Synthetic audio
synthetic_probability ≈ 0.7552

Strong synthetic example
synthetic_probability ≈ 0.9997
```

These examples are development observations and **must not be interpreted as formal accuracy measurements**.

---

# ⚠️ Current Limitations

The current implementation is a **prototype** and has several limitations.

### 1. Limited audio window

The current main detector processes approximately:

```text
4.04 seconds
```

of an input recording.

For long recordings, only the initial window is currently analyzed.

---

### 2. Short audio handling

Audio shorter than the required model input is repeated to reach the required length.

This is useful for prototype testing but requires further validation for production use.

---

### 3. No formal threshold calibration

The current threshold:

```text
0.5
```

is not yet calibrated against a representative validation dataset.

---

### 4. No probability calibration

The returned `synthetic_probability` is derived from the model's softmax output.

It should therefore be treated as a **model score/probability estimate**, not as a calibrated real-world probability of voice cloning.

---

### 5. No formal production metrics yet

Formal evaluation such as:

```text
Accuracy
Precision
Recall
F1-score
EER
FAR
FRR
ROC-AUC
```

has not yet been established for the project's target deployment conditions.

---

### 6. Dataset limitation

The pretrained AASIST model was developed and evaluated in the context of audio anti-spoofing datasets, particularly **ASVspoof**.

Real-world voice cloning attacks can differ significantly from benchmark datasets.

Additional validation against modern voice cloning systems and real-world audio conditions is therefore required.

---

# 🛣️ Future Development

The planned development of the synthetic voice detection branch includes:

```text
Current
  ↓
Single-window AASIST
  ↓
Voice Activity Detection
  ↓
Multi-window analysis
  ↓
Window-level score aggregation
  ↓
Threshold calibration
  ↓
Validation on diverse synthetic voices
  ↓
Robustness testing
  ↓
Real-time inference optimization
  ↓
Integration with the overall security pipeline
```

The final system is intended to provide a synthetic voice score that can be consumed by the project's downstream risk-analysis system.

---

# 🏗️ Overall SIH System

The broader project architecture is:

```text
                    ┌─────────────────────┐
                    │      Audio Input    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Audio Preprocessing │
                    │       + VAD         │
                    └──────────┬──────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │       AI Analysis Layer       │
               │                               │
               │   Branch 1: AASIST            │
               │   Synthetic Voice Detection   │
               │                               │
               │   Future Detection Models     │
               └───────────────┬───────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature / Score     │
                    │ Fusion               │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Risk Engine      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Security Policy     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Android        │
                    └─────────────────────┘
```

> **Current repository scope:** Branch 1 — Synthetic Voice Detection.

---

# 📚 AASIST Reference

The AASIST implementation included in this repository is based on the original open-source project by NAVER Corp.

Original repository:

https://github.com/clovaai/aasist

Research paper:

https://arxiv.org/abs/2110.01200

### Citation

```bibtex
@INPROCEEDINGS{Jung2021AASIST,
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin
          and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks},
  year={2021}
}
```

---

# 📜 License

The original AASIST implementation is distributed under its respective open-source license.

Please see:

```text
aasist/LICENSE
aasist/NOTICE
```

for the applicable copyright and licensing information.

---

# 👨‍💻 Project

**Smart India Hackathon 2026**

### Problem Area

**AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks**

### Current Module

**Synthetic Voice Detection — AASIST**

---

## ⭐ Repository Status

```text
Prototype Development
       ↓
AASIST Integration        ✅
Pretrained Model           ✅
CPU Inference              ✅
Synthetic Detection        ✅
Basic Testing              ✅
GitHub Integration         ✅

VAD                       🔄 Planned
Multi-window Detection    🔄 Planned
Threshold Calibration    🔄 Planned
Robust Evaluation         🔄 Planned
Real-time Pipeline        🔄 Planned
API Integration           🔄 Planned
Android Integration       🔄 Planned
```
