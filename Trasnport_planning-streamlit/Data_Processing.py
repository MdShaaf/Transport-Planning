import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging

# Setting up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('Data_Processing')
logger.setLevel(logging.DEBUG)  # ✅ FIX: Use logging.DEBUG instead of string

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # ✅ FIX: Use logging.DEBUG

log_file_path = os.path.join(log_dir, 'Data_Processing.log')  # ✅ FIX: Add .log extension
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)  # ✅ FIX: Use logging.DEBUG

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("Reading the file....")

def load_data(parquet_path, output_dir):
    try:
        file_name = os.path.basename(parquet_path)
        month_tag = file_name.replace("yellow_tripdata_", "").replace(".parquet", "")
        df = pd.read_parquet(parquet_path)

        logger.info(f"{file_name} Data loaded successfully with shape: {df.shape}")
        logger.info("Creating the stratify variable for sampling...")
        
        # Create a stratify variable based on 'passenger_count' for stratified sampling
        df['Hour'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.hour
        df['Day of Week'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.dayofweek

        # Combine 'Hour' and 'Day of Week' to create a stratify variable
        df['Stratify_Var'] = df['Hour'].astype(str) + '_' + df['Day of Week'].astype(str)

        # ✅ IMPORTANT: remove rare strata
        df = df.groupby('Stratify_Var').filter(lambda x: len(x) > 1)

        logger.info("Stratify variable created successfully.")
        df_train, df_sampled = train_test_split(
            df, test_size=0.1, random_state=42, stratify=df['Stratify_Var']
        )

        df = df_sampled.drop(columns=['Hour', 'Day of Week', 'Stratify_Var'])

        logger.info(f"Data sampled successfully with new shape: {df.shape}")
        output_file = os.path.join(output_dir, f"{month_tag}_sampled_data.parquet")
        df.to_parquet(output_file, index=False)
        logger.info(f"Sampled data saved to '{output_file}'.")

        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# Directory setup
output_dir = r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\Transport Planning\Sampled_Data"
os.makedirs(output_dir, exist_ok=True)

DATA_DIR = r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\Transport Planning\Data"
parquet_files = [
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".parquet") and f.startswith("yellow_tripdata_")
]
all_df=[]
# ✅ FIX: Loop through all parquet files
for parquet_file in parquet_files:
    parquet_path = os.path.join(DATA_DIR, parquet_file)
    df = load_data(parquet_path, output_dir)
    all_df.append(df)

# Combine all sampled dataframes into one
combined_df = pd.concat(all_df, ignore_index=True, axis=0)
combined_output_file = os.path.join(output_dir, "combined_sampled_data.parquet")
combined_df.to_parquet(combined_output_file, index=False)
logger.info(f"Combined sampled data saved to '{combined_output_file}' with shape: {combined_df.shape}")