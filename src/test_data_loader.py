from utils.data_loader import load_ipinyou_logs

df = load_ipinyou_logs("D:/Research Methodology/DAAWBC/dynamic_ad_allocation/data/ipinyou")

print("\n✅ Data loaded successfully!\n")
print(df.head())
print("\nColumns:", df.columns.tolist())
print("Rows:", len(df))
