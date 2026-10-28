import pandas as pd

df = pd.read_csv("data/train-test.csv")

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nDATA TYPES:")
print(df.dtypes)

print("\nMISSING VALUES:")
print(df.isnull().sum())   

print("\nDUPLICATE LOAD IDS:")
print(df["load_id"].duplicated().sum())

print("\nSTATISTICAL SUMMARY:")
print(df.describe())   

print("\nFREIGHT RATE STATISTICS:")
print(df["posted_rate"].describe())   