import pandas as pd
import numpy as np
from pathlib import Path

def load_ipinyou_logs(path: str) -> pd.DataFrame:
    """
    Load iPinYou logs from either:
    - a folder containing .txt/.csv files, OR
    - a single .txt/.csv file.

    Required columns:
      - click
      - market_price (or payprice -> renamed)
    """
    p = Path(path)

    # 1) Collect files
    if p.is_file():
        files = [p]
    else:
        files = sorted(p.glob("*.txt"))
        if not files:
            files = sorted(p.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No dataset files found in: {p}")

    # 2) Read + concat
    dfs = []
    for f in files:
        print(f"Loading {f.name} ...")
        sep = "\t" if f.suffix.lower() == ".txt" else ","
        df = pd.read_csv(f, sep=sep, engine="python")
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(data)}")

    # 3) Normalize column names
    rename_map = {
        "payprice": "market_price",
        "slotwidth": "slot_w",
        "slotheight": "slot_h",
    }
    for old, new in rename_map.items():
        if old in data.columns:
            data = data.rename(columns={old: new})

    # 4) Validate required columns
    if "market_price" not in data.columns or "click" not in data.columns:
        raise ValueError("Dataset missing required columns: 'market_price' and/or 'click'.")

    # 5) Clean + enforce basic types
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["market_price", "click"])
    data["market_price"] = data["market_price"].astype(float)
    data["click"] = data["click"].astype(int).clip(0, 1)

    return data
