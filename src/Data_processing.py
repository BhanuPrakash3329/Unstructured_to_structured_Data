import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = r"C:\Users\bhanu\OneDrive\Desktop\Assignment-1\raw\synthetic_bmi_data.csv"
df = pd.read_csv(file_path)

# Display basic information
df.info()
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Check for duplicate rows
duplicates = df.duplicated().sum()
print("Duplicate Rows:", duplicates)

# Drop duplicates if any
df = df.drop_duplicates()

# Fill missing values (if any) using median strategy
df.fillna(df.median(numeric_only=True), inplace=True)

# Scale numerical features using StandardScaler
scaler = StandardScaler()
numerical_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                      'DiabetesPedigreeFunction', 'Age']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Check for categorical features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical Features:", categorical_features)

# One-hot encoding (if needed)
if categorical_features:
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Split dataset into features and target
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Display shapes
print("Training Data Shape:", X_train.shape, y_train.shape)
print("Testing Data Shape:", X_test.shape, y_test.shape)

# Save the processed dataset
processed_file_path = r"C:\Users\bhanu\OneDrive\Desktop\Assignment-1\Processed_Data\processed_synthetic_bmi_data.csv"
df.to_csv(processed_file_path, index=False)
