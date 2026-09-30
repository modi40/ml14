import pandas as pd

df = pd.read_csv("student_scores.csv")

print("Created Dataset with Null Values Loaded from CSV:")
print(df)

print("\nDetected Null Values Per Column:")
print(df.isnull().sum())
print("Total Null Values Count:", df.isnull().sum().sum())

cleaned_df = df.dropna()
print("\nCleaned Dataset After Removing Null Values:")
print(cleaned_df)
print("\nNull Values in Cleaned Dataset:")
print(cleaned_df.isnull().sum())
