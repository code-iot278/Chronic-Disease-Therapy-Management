# =========================================
# MIN-MAX NORMALIZATION - FULL CODE
# =========================================

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------
# 1. Load input CSV file
# -----------------------------------------
input_file = "data.csv"
df = pd.read_csv(input_file)

print("Original Data:")
print(df)

# -----------------------------------------
# 2. Initialize Min-Max Scaler (0 to 1)
# -----------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))

# -----------------------------------------
# 3. Apply normalization
# -----------------------------------------
normalized_data = scaler.fit_transform(df)

# -----------------------------------------
# 4. Convert back to DataFrame
# -----------------------------------------
df_normalized = pd.DataFrame(
    normalized_data,
    columns=df.columns
)

print("\nNormalized Data:")
print(df_normalized)

# -----------------------------------------
# 5. Save normalized data to CSV
# -----------------------------------------
output_file = "normalized_data.csv"
df_normalized.to_csv(output_file, index=False)

print(f"\n✅ Normalized data saved as '{output_file}'")
