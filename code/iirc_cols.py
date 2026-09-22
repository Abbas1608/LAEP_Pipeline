import pandas as pd

df = pd.read_csv("F2_combined_dataset.csv")

# Fill NaN values with median or mean of available IIRS pixels
iirs_cols = [
    "Reflectance_IIRS_1500nm",
    "Reflectance_IIRS_2000nm",
    "IIRS_Band_Ratio",
    "IIRS_H2O_Absorption_Depth",
]
for col in iirs_cols:
    df[col] = df[col].fillna(df[col].median())

df.to_csv("F2_combined_dataset_imputed.csv", index=False)