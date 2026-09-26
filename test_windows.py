import soundfile as sf
import numpy as np
import torch
import time

from synthetic_detection import model, DEVICE, TARGET_SR


AUDIO_FILE = r"D:\M.Tech\VOICE_CLONING_DETECTOR\aasist\long_speech.wav"

WINDOW_SECONDS = 4
WINDOW_SAMPLES = int(WINDOW_SECONDS * TARGET_SR)


# ============================================================
# READ AUDIO
# ============================================================

audio, sample_rate = sf.read(AUDIO_FILE)

print("Original sample rate:", sample_rate)

# Stereo → Mono
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

audio = audio.astype(np.float32)


# ============================================================
# RESAMPLE
# ============================================================

if sample_rate != TARGET_SR:

    audio_tensor = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0)

    audio_tensor = torch.nn.functional.interpolate(
        audio_tensor.unsqueeze(0),
        size=int(len(audio) * TARGET_SR / sample_rate),
        mode="linear",
        align_corners=False
    ).squeeze(0)

    audio = audio_tensor.squeeze(0).numpy()


# ============================================================
# SELECT WINDOWS
# ============================================================

duration = len(audio) / TARGET_SR

print("Duration:", round(duration, 2), "seconds")
print()


# Test windows at these positions
start_times = [0, 10, 20, 30, 40, 50]


print("==============================")
print("      WINDOW ANALYSIS")
print("==============================")


for start_second in start_times:

    start_sample = int(
        start_second * TARGET_SR
    )

    end_sample = (
        start_sample +
        WINDOW_SAMPLES
    )

    # Make sure window exists
    if end_sample > len(audio):
        print(
            f"{start_second}s window: skipped"
        )
        continue

    window = audio[
        start_sample:end_sample
    ]

    # Tensor
    audio_tensor = torch.tensor(
        window,
        dtype=torch.float32
    ).unsqueeze(0)

    audio_tensor = audio_tensor.to(DEVICE)

    # Inference
    start_time = time.perf_counter()

    with torch.no_grad():

        _, output = model(
            audio_tensor
        )

    end_time = time.perf_counter()

    # Softmax
    probabilities = torch.softmax(
        output,
        dim=1
    )

    spoof_probability = (
        probabilities[0][0].item()
    )

    bonafide_probability = (
        probabilities[0][1].item()
    )

    print(
        f"{start_second:02d}-{start_second + WINDOW_SECONDS:02d} sec"
        f" | Synthetic: {spoof_probability:.4f}"
        f" | Bonafide: {bonafide_probability:.4f}"
        f" | Time: {end_time - start_time:.4f}s"
    )


print("==============================")