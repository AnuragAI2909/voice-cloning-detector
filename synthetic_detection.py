import os
import json
import time

import torch
import torchaudio
import soundfile as sf
import numpy as np

from aasist.models.AASIST import Model


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AASIST_DIR = os.path.join(
    BASE_DIR,
    "aasist"
)

CONFIG_PATH = os.path.join(
    AASIST_DIR,
    "config",
    "AASIST.conf"
)

CHECKPOINT_PATH = os.path.join(
    AASIST_DIR,
    "models",
    "weights",
    "AASIST.pth"
)


# ============================================================
# AASIST SETTINGS
# ============================================================

TARGET_SAMPLE_RATE = 16000

MODEL_INPUT_SAMPLES = 64600

DEVICE = torch.device("cpu")


# ============================================================
# LOAD AASIST CONFIGURATION
# ============================================================

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)


# ============================================================
# LOAD AASIST MODEL
# ============================================================

print("Loading AASIST model...")

model = Model(config["model_config"])

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.to(DEVICE)

model.eval()

print("AASIST model loaded successfully!")


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def preprocess_audio(audio_file):
    """
    Prepare an audio file for AASIST inference.

    Steps:
    1. Load audio
    2. Convert stereo to mono
    3. Convert to float32
    4. Resample to 16 kHz
    5. Prepare exactly 64,600 samples
    6. Add batch dimension
    """

    # Load audio
    audio, sample_rate = sf.read(audio_file)

    # --------------------------------------------------------
    # Stereo → Mono
    # --------------------------------------------------------

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    audio = audio.astype(np.float32)


    # --------------------------------------------------------
# Validate audio
# --------------------------------------------------------

    if len(audio) == 0:
        raise ValueError(
            "Audio file contains no samples."
        )


    # --------------------------------------------------------
    # Resample → 16 kHz
    # --------------------------------------------------------

    if sample_rate != TARGET_SAMPLE_RATE:

        audio_tensor = torch.tensor(
            audio,
            dtype=torch.float32
        )

        audio_tensor = audio_tensor.unsqueeze(0)

        audio_tensor = torchaudio.functional.resample(
            audio_tensor,
            sample_rate,
            TARGET_SAMPLE_RATE
        )

        audio = audio_tensor.squeeze(0).numpy()

    # --------------------------------------------------------
    # Prepare model input length
    # --------------------------------------------------------

    if len(audio) >= MODEL_INPUT_SAMPLES:

        # Keep first ~4.04 seconds
        audio = audio[:MODEL_INPUT_SAMPLES]

    else:

        # Repeat short audio until required length
        repeats = int(
            np.ceil(
                MODEL_INPUT_SAMPLES / len(audio)
            )
        )

        audio = np.tile(
            audio,
            repeats
        )

        audio = audio[:MODEL_INPUT_SAMPLES]

    # --------------------------------------------------------
    # Convert to PyTorch tensor
    # --------------------------------------------------------

    audio_tensor = torch.tensor(
        audio,
        dtype=torch.float32
    )

    # Add batch dimension
    audio_tensor = audio_tensor.unsqueeze(0)

    return audio_tensor


# ============================================================
# SYNTHETIC VOICE ANALYSIS
# ============================================================

def analyze_synthetic_voice(audio_file):
    """
    Analyze an audio file using pretrained AASIST.

    Returns:
        dict containing:
        - synthetic_probability
        - bonafide_probability
        - inference_time_seconds
        - result
    """

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not os.path.exists(audio_file):

        raise FileNotFoundError(
            f"Audio file not found: {audio_file}"
        )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    audio = preprocess_audio(audio_file)

    audio = audio.to(DEVICE)

    # --------------------------------------------------------
    # AASIST inference
    # --------------------------------------------------------

    start_time = time.perf_counter()

    with torch.no_grad():

        _, output = model(audio)

    end_time = time.perf_counter()

    inference_time = end_time - start_time

    # --------------------------------------------------------
    # Convert model output to probabilities
    # --------------------------------------------------------

    probabilities = torch.softmax(
        output,
        dim=1
    )

    # AASIST:
    # Class 0 = Spoof
    # Class 1 = Bonafide

    spoof_probability = probabilities[0][0].item()

    bonafide_probability = probabilities[0][1].item()

    # Our backend-facing synthetic score
    synthetic_probability = spoof_probability

    # --------------------------------------------------------
    # POC decision
    # --------------------------------------------------------

    if synthetic_probability >= 0.5:

        result = "SYNTHETIC / SPOOF"

    else:

        result = "BONAFIDE / REAL"

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "synthetic_probability":
            round(synthetic_probability, 4),

        "bonafide_probability":
            round(bonafide_probability, 4),

        "inference_time_seconds":
            round(inference_time, 4),

        "result":
            result
    }