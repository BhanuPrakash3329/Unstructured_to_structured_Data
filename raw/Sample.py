# Re-run the dataset generation as execution state was reset

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data with 1000 rows
data = {
    "Pregnancies": np.random.randint(0, 10, 1000),
    "Glucose": np.random.randint(50, 200, 1000),
    "BloodPressure": np.random.randint(40, 120, 1000),
    "SkinThickness": np.random.randint(10, 50, 1000),
    "Insulin": np.random.randint(0, 300, 1000),
    "BMI": np.round(np.random.uniform(15, 45, 1000), 1),
    "DiabetesPedigreeFunction": np.round(np.random.uniform(0.1, 2.5, 1000), 3),
    "Age": np.random.randint(18, 80, 1000),
    "Outcome": np.random.randint(0, 2, 1000)  # 0: No diabetes, 1: Diabetes
}

# Create DataFrame
df_synthetic = pd.DataFrame(data)

# Save as CSV
synthetic_file_path = "C:/Users/bhanu/OneDrive/Desktop/Assignment-1/raw/synthetic_bmi_data.csv"
df_synthetic.to_csv(synthetic_file_path, index=False)

# Return the file path
synthetic_file_path
