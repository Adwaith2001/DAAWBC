import os
import sys
from datetime import datetime
import pandas as pd
from pathlib import Path

CAMPAIGN_IDS   = ['3427', '3476']
MAX_ROWS       = None
ROOT_INPUT_DIR = Path("D:/dataset/ipinyou-project/make-ipinyou-data/filtered_output")
LOG_FILES      = ["train.log.txt", "test.log.txt"]


def get_device_type(useragent: str) -> int:
    ua = useragent.lower()
    if any(x in ua for x in ['ipad', 'tablet', 'kindle']):
        return 2
    elif any(x in ua for x in ['mobile', 'android', 'iphone',
                                 'ipod', 'windows phone', 'blackberry']):
        return 1
    else:
        return 0


def get_usertag_count(usertag: str) -> int:
    if not usertag or usertag.strip() == '' or usertag.strip() == 'null':
        return 0
    return len(usertag.strip().split(','))


def _parse_log_line(parts):
    if len(parts) < 25:
        return None
    try:
        ts           = parts[1]
        hour         = int(ts[8:10])
        dt           = datetime.strptime(ts[:8], "%Y%m%d")
        weekday      = dt.weekday()
        siteid       = parts[9]
        slot_w       = int(parts[13])
        slot_h       = int(parts[14])
        market_price = float(parts[20])
        click        = int(parts[24])
        region          = int(parts[6])  if parts[6].strip().isdigit()  else 0
        slotvisibility  = int(parts[15]) if parts[15].strip().isdigit() else 0
        slotformat      = int(parts[16]) if parts[16].strip().isdigit() else 0
        device_type     = get_device_type(parts[4])
        usertag_count   = get_usertag_count(parts[23] if len(parts) > 23 else '')
    except (ValueError, IndexError):
        return None

    return {
        "click":          click,
        "market_price":   market_price,
        "weekday":        weekday,
        "hour":           hour,
        "siteid":         siteid,
        "slot_w":         slot_w,
        "slot_h":         slot_h,
        "region":         region,
        "slotvisibility": slotvisibility,
        "slotformat":     slotformat,
        "device_type":    device_type,
        "usertag_count":  usertag_count,
    }


def transform_logs(campaign_dir, max_rows=None):
    all_records = []
    for log_file in LOG_FILES:
        file_path = campaign_dir / log_file
        if not file_path.exists():
            print(f"  !! {file_path.name} not found. Skipping.")
            continue
        print(f"  -> Parsing {log_file}...")
        with open(file_path, "r", encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                parts  = line.rstrip("\n").split("\t")
                record = _parse_log_line(parts)
                if record is not None:
                    all_records.append(record)
    df = pd.DataFrame(all_records)
    print(f"  Total rows: {len(df):,}")
    return df


def main():
    for campaign_id in CAMPAIGN_IDS:
        campaign_dir = ROOT_INPUT_DIR / campaign_id
        if not campaign_dir.exists():
            print(f"Skipping {campaign_id}: folder not found.")
            continue

        print(f"\n{'='*55}")
        print(f" Processing Campaign: {campaign_id}")
        print(f"{'='*55}")

        out_dir = campaign_dir / "enhanced"
        out_dir.mkdir(exist_ok=True)

        df = transform_logs(campaign_dir, MAX_ROWS)

        if len(df) == 0:
            print(f"  ERROR: No data. Skipping.")
            continue

        df.dropna(subset=["click", "market_price",
                           "slot_w", "slot_h"], inplace=True)
        df["siteid"] = df["siteid"].fillna("UNKNOWN_SITE")

        out_file = out_dir / "final_sample_log_enhanced.txt"
        df.to_csv(out_file, sep="\t", index=False)

        print(f"✅ Saved: {out_file}")
        print(f"  Shape : {df.shape}")
        print(f"  CTR   : {df['click'].mean()*100:.4f}%")
        print(f"  Price : {df['market_price'].mean():.1f}")

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()