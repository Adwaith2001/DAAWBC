from utils.data_loader import load_ipinyou_logs
from utils.pctr import fit_pctr_model, add_pctr

DATA_DIR = r"D:/Research Methodology/DAAWBC/dynamic_ad_allocation/data/ipinyou"

df = load_ipinyou_logs(DATA_DIR)

m = max(1, int(0.75 * len(df)))
train_df, test_df = df.iloc[:m].reset_index(drop=True), df.iloc[m:].reset_index(drop=True)

model, use_cols = fit_pctr_model(train_df)
test_df = add_pctr(test_df, model, use_cols)

print("✅ pCTR added. Preview:")
print(test_df.head())
print("\nColumns:", test_df.columns.tolist())
