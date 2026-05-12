import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "sample_log_with_pctr.txt")

df = pd.read_csv(DATA_PATH, sep="\t")

print("Total rows:", len(df))
print("Total clicks:", df["click"].sum())
print("CTR:", df["click"].mean())
