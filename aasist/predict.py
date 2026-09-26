import torch  # pyright: ignore[reportMissingImports]
import torchaudio  # pyright: ignore[reportMissingImports]
import soundfile as sf  # pyright: ignore[reportMissingImports]
import numpy as np
import json
from models.AASIST import Model


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

AUDIO_FILE = "real_short.wav"
MODEL_FILE = "models/weights/AASIST.pth"
CONFIG_FILE = "config/AASIST.conf"

MAX_LEN = 64600


# --------------------------------------------------
# LOAD CONFIG
# --------------------------------------------------

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)


model_config = config["model_config"]


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device("cpu")

print("Device:", device)


# --------------------------------------------------
# CREATE MODEL
# --------------------------------------------------

model = Model(
    model_config)


# --------------------------------------------------
# LOAD PRETRAINED WEIGHTS
# --------------------------------------------------

print("Loading AASIST model...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("Model loaded successfully!")


# --------------------------------------------------
# LOAD AUDIO
# --------------------------------------------------

print("Loading audio:", AUDIO_FILE)

audio, sample_rate = sf.read(AUDIO_FILE)

print("Original sample rate:", sample_rate)
print("Original samples:", len(audio))


# --------------------------------------------------
# CONVERT STEREO → MONO
# --------------------------------------------------

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# --------------------------------------------------
# RESAMPLE → 16 kHz
# --------------------------------------------------

TARGET_SR = 16000

if sample_rate != TARGET_SR:

    audio_tensor = torch.tensor(
        audio,
        dtype=torch.float32
    )

    audio_tensor = audio_tensor.unsqueeze(0)

    audio_tensor = torchaudio.functional.resample(
        audio_tensor,
        sample_rate,
        TARGET_SR
    )

    audio = audio_tensor.squeeze(0).numpy()

    sample_rate = TARGET_SR

print("Final sample rate:", sample_rate)
print("Samples after resampling:", len(audio))


# --------------------------------------------------
# MAKE AUDIO EXACTLY 64600 SAMPLES
# --------------------------------------------------

if len(audio) >= MAX_LEN:

    audio = audio[:MAX_LEN]

else:

    repeats = int(np.ceil(MAX_LEN / len(audio)))

    audio = np.tile(audio, repeats)

    audio = audio[:MAX_LEN]


print("Final samples:", len(audio))


# --------------------------------------------------
# CONVERT TO PYTORCH TENSOR
# --------------------------------------------------

audio_tensor = torch.tensor(
    audio,
    dtype=torch.float32
)

# Add batch dimension
audio_tensor = audio_tensor.unsqueeze(0)

audio_tensor = audio_tensor.to(device)


# --------------------------------------------------
# RUN AASIST
# --------------------------------------------------

print("Running AASIST...")

with torch.no_grad():

    hidden, output = model(audio_tensor)


# --------------------------------------------------
# GET CLASS SCORES
# --------------------------------------------------

print("\nRaw model output:")
print(output)


# Convert logits → probabilities
probabilities = torch.softmax(output, dim=1)

bonafide_probability = probabilities[0][1].item()
spoof_probability = probabilities[0][0].item()


# --------------------------------------------------
# FINAL RESULT
# --------------------------------------------------

print("\n==============================")
print("      VOICE ANALYSIS")
print("==============================")

print(
    f"BONAFIDE probability : {bonafide_probability * 100:.2f}%"
)

print(
    f"SPOOF probability    : {spoof_probability * 100:.2f}%"
)


if spoof_probability > bonafide_probability:

    print("\nRESULT: 🚨 SPOOF / FAKE VOICE")

else:

    print("\nRESULT: ✅ BONAFIDE / REAL VOICE")

print("==============================")