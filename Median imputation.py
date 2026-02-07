import pandas as pd

# ---------------------------------
# 1. Read input CSV file
# ---------------------------------
df = pd.read_csv("data.csv")

print("Original Data:")
print(df)

# ---------------------------------
# 2. Apply Median Imputation
#    Replace missing values with column median
# ---------------------------------
df_imputed = df.fillna(df.median())

print("\nAfter Median Imputation:")
print(df_imputed)

# ---------------------------------
# 3. Save output CSV file
# ---------------------------------
df_imputed.to_csv("median_imputed_data.csv", index=False)

print("\n✅ Median imputation completed and saved.")
