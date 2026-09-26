import time

from synthetic_detection import preprocess_audio, model, DEVICE


AUDIO_FILE = r"aasist\real_short.wav"

NUM_RUNS = 10


# -----------------------------
# Prepare audio once
# -----------------------------

audio = preprocess_audio(AUDIO_FILE)
audio = audio.to(DEVICE)


# -----------------------------
# Warm-up
# -----------------------------

print("Running warm-up...")

with __import__("torch").no_grad():
    for _ in range(3):
        model(audio)


# -----------------------------
# Benchmark
# -----------------------------

times = []

print("Running benchmark...")

for i in range(NUM_RUNS):

    start = time.perf_counter()

    with __import__("torch").no_grad():
        model(audio)

    end = time.perf_counter()

    elapsed = end - start

    times.append(elapsed)

    print(
        f"Run {i + 1}: {elapsed:.4f} seconds"
    )


# -----------------------------
# Results
# -----------------------------

average_time = sum(times) / len(times)
minimum_time = min(times)
maximum_time = max(times)


print("\n==============================")
print("      BENCHMARK RESULTS")
print("==============================")

print(
    f"Average inference time : {average_time:.4f} seconds"
)

print(
    f"Minimum inference time : {minimum_time:.4f} seconds"
)

print(
    f"Maximum inference time : {maximum_time:.4f} seconds"
)

print("==============================")