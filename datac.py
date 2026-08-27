import pandas as pd

# Load A–M and N–Z CSVs
df1 = pd.read_csv("group1_A_M_dataset.csv", header=None)
df2 = pd.read_csv("group2_N_Z_dataset.csv", header=None)

# Combine and shuffle
df_combined = pd.concat([df1, df2], ignore_index=True).sample(frac=1, random_state=42)

# Save merged dataset
df_combined.to_csv("full_A_Z_dataset.csv", index=False, header=False)
print("✅ Combined A–Z dataset saved as full_A_Z_dataset.csv")
