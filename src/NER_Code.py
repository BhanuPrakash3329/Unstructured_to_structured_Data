import pandas as pd
import spacy
import re

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load dataset
csv_path = r"C:\Users\bhanu\OneDrive\Desktop\Assignment-1\raw\synthetic_bmi_data.csv"
df = pd.read_csv(csv_path)

# Convert to long format for easier analysis
df_long = df.melt(var_name="parameter", value_name="value")

# Combine columns into text format for NER and regex
df_long["text"] = df_long["parameter"].astype(str) + " " + df_long["value"].astype(str)

# Extract named entities using spaCy
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

df_long["entities"] = df_long["text"].apply(extract_entities)

# Use regex to extract numeric medical values
def extract_numeric(text):
    pattern = r"(BMI|Weight|Height|BP|Blood Pressure)[\s:]*([\d.]+)"
    match = re.findall(pattern, text, flags=re.IGNORECASE)
    return match if match else None

df_long["regex_extracted"] = df_long["text"].apply(extract_numeric)

# Show sample output
print(df_long.head(10))

# Optional: Save output to CSV for inspection
df_long.to_csv("bmi_named_entity_output.csv", index=False)
