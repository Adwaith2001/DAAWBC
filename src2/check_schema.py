import pandas as pd

# Check train.log.txt columns
df = pd.read_csv(
    'D:/dataset/ipinyou-project/make-ipinyou-data/filtered_output/1458/train.log.txt',
    sep='\t',
    header=None,
    nrows=5
)

print("Number of columns:", df.shape[1])
print("\nFirst 5 rows:")
print(df.to_string())

# Also check schema.txt
print("\n--- schema.txt ---")
with open('D:/dataset/ipinyou-project/make-ipinyou-data/schema.txt', 'r') as f:
    print(f.read())
    