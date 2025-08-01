import pandas as pd

# Load the processed data
df_long = pd.read_csv("bmi_named_entity_output.csv")

# 1. Validation: Check for missing or invalid values
print("\n🔍 Checking for missing values:")
print(df_long.isnull().sum())

# 2. Validation: Check if 'value' column contains only numeric data
print("\n🔍 Checking non-numeric values in 'value' column:")
non_numeric = df_long[pd.to_numeric(df_long['value'], errors='coerce').isnull()]
print(non_numeric[['parameter', 'value']])

# 3. Validation: Check if NER returned any entities
print("\n🔍 Sample of extracted entities:")
print(df_long[df_long['entities'].notnull()][['text', 'entities']].head(10))

# 4. Validation: Check regex extraction output
print("\n🔍 Sample of regex extracted values:")
print(df_long[df_long['regex_extracted'].notnull()][['text', 'regex_extracted']].head(10))

# 5. Summary counts
print("\n✅ Summary:")
print(f"Total rows: {len(df_long)}")
print(f"Rows with NER entities: {df_long['entities'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0).astype(bool).sum()}")
print(f"Rows with regex matches: {df_long['regex_extracted'].notnull().sum()}")
