from synthetic_detection import analyze_synthetic_voice


result = analyze_synthetic_voice(
     r"D:\M.Tech\VOICE_CLONING_DETECTOR\aasist\long_speech.wav")


print("\n==============================")
print("      VOICE ANALYSIS")
print("==============================")

print(
    "Synthetic probability :",
    result["synthetic_probability"]
)

print(
    "Bonafide probability  :",
    result["bonafide_probability"]
)

print(
    "Inference time        :",
    result["inference_time_seconds"],
    "seconds"
)

print(
    "Result                :",
    result["result"]
)

print("==============================")