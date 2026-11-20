import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent

for name in ("test_prob", "baseline_test_prob", "y_test_true"):
    arr = np.load(HERE / f"{name}.npy")
    print(f"\n{name}:")
    print("Shape:", arr.shape)
    print("Dtype:", arr.dtype)
    print(arr.tolist())
