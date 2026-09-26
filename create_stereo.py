import soundfile as sf
import numpy as np

# Input audio
input_file = r"aasist\real_short.wav"

# Output stereo audio
output_file = r"aasist\real_stereo.wav"

# Read original audio
audio, sample_rate = sf.read(input_file)

# Make sure audio is mono first
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Create stereo:
# Left channel = original
# Right channel = original
stereo_audio = np.column_stack((audio, audio))

# Save stereo WAV
sf.write(
    output_file,
    stereo_audio,
    sample_rate
)

print("Stereo audio created successfully!")
print("Sample rate:", sample_rate)
print("Channels: 2")
print("Output:", output_file)