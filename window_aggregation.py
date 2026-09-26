import numpy as np


# Scores from REAL recording
real_scores = [
    0.9995,
    0.9562,
    0.0059,
    0.7075,
    0.9994,
    0.1719
]


# Scores from SYNTHETIC recording
synthetic_scores = [
    0.9994,
    1.0000,
    0.9981,
    0.9999,
    1.0000,
    0.9998
]


def analyze_scores(name, scores):

    mean_score = np.mean(scores)

    median_score = np.median(scores)

    max_score = np.max(scores)

    print("\n==============================")
    print(name)
    print("==============================")

    print(
        "Mean score   :",
        round(mean_score, 4)
    )

    print(
        "Median score :",
        round(median_score, 4)
    )

    print(
        "Maximum score:",
        round(max_score, 4)
    )


analyze_scores(
    "REAL RECORDING",
    real_scores
)


analyze_scores(
    "SYNTHETIC RECORDING",
    synthetic_scores
)