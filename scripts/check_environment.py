import sys
import platform

import numpy as np
import pandas as pd
import sklearn
import lightgbm
import shap

print("=" * 60)
print("AIRGUARD-LK ENVIRONMENT CHECK")
print("=" * 60)

print("Python:", sys.version)
print("Platform:", platform.platform())

print()
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("LightGBM:", lightgbm.__version__)
print("SHAP:", shap.__version__)

print()
print("Environment ready.")
