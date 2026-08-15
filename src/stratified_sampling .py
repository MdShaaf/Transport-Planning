import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
import os
import logging
from pathlib import Path
# Setting up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

project_dir = Path()
data_dir = project_dir/"Data"

# logging configuration
logger = logging.getLogger('Data_sampling')
logger.setLevel(logging.DEBUG) 

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  

log_file_path = os.path.join(log_dir, 'Data_sampling.log') 
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)  

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("Reading the file....")

###Zone Id File for Borough Wise data split
try:
    logger.info(f"reading location ID Files")
    location_ids= pd.read_csv(data_dir/"location_ids.csv")
except Exception as e:
    logger.error(f"Error while reading the Location ID File {e}")
    raise



def preporcess_create_hourly_borough_data(parquet_path, output_dir):
    try:
        file_name = os.path.basename(parquet_path)
        month_tag = file_name.replace("yellow_tripdata_", "").replace(".parquet", "")
        data = pd.read_parquet(parquet_path)

        logger.info(f"{file_name} Data loaded successfully with shape: {data.shape}")
        data = data.merge(location_ids, how="left", left_on='PULocationID', right_on='LocationID')
        boroughs = ['Manhattan', 'Queens', 'Brooklyn', 'Bronx']
        data = data[data['Borough'].isin(boroughs)].copy()
        data['tpep_pickup_datetime']=pd.to_datetime(data['tpep_pickup_datetime'])

        ######
        #Data Cleaning
        ##----------------------

        pickup_location = {"Newark Airport":2,"JFK Airport":3} 
        based_on_drop_location = {132:3,1:2}
        data['RatecodeID']=  data['RatecodeID'].fillna(data['DOLocationID'].map(based_on_drop_location))
        data['RatecodeID']=  data['RatecodeID'].fillna(data['Zone'].map(pickup_location)) #####{"Newark Airport":2,"JFK Airport":3}
        ##Rest all the RateCodeID becomes 1
        data['RatecodeID']=data['RatecodeID'].fillna(1)

        ##Creating bins for the distance travelled
        bins= [0, 1, 3, 5, 10, 20, 50]
        data=data[data['trip_distance']<=50].copy() ## found about 500 values where the distance is beyond 50miles, dropping them as outlier
        data['Trip Distance Bins'] = pd.cut(data['trip_distance'], 
                                            bins=bins,labels=['0-1','1-3','3-5','5-10','10-20','20-50'], include_lowest=True)
        
        ##There are NAN values in passenger counts,Most of the values are 1 based on analysis and review
        data['passenger_count']=data['passenger_count'].fillna(1)

        ##there are outliers with 6, 7 and 8 passenger counts
        data =data[data['passenger_count']<=5]
        data['congestion_surcharge']=data['congestion_surcharge'].fillna(0)
        data['store_and_fwd_flag']=data['store_and_fwd_flag'].fillna('N')
        data['Airport_fee']=data['Airport_fee'].fillna(0)
        logger.info(f"data shape after processing {data.shape}")

        hourly = data.groupby(['Borough', pd.Grouper(key='tpep_pickup_datetime', freq='h')]).size().reset_index(name='Trips')
        logger.info(f"Data sampled successfully with new shape: {hourly.shape}")
        # output_file = os.path.join(output_dir, f"{month_tag}_sampled_data.parquet")
        # hourly.to_csv(output_file, index=False)
        # logger.info(f"Sampled data saved to '{output_file}'.")
        return hourly
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# Directory setup
output_dir = project_dir / "Data/Stratified_sampled_data"
os.makedirs(output_dir, exist_ok=True)


parquet_files = [
    f for f in os.listdir(data_dir)
    if f.endswith(".parquet") and f.startswith("yellow_tripdata_")
]
all_df=[]

# Loop through all parquet files
for parquet_file in parquet_files:
    parquet_path = data_dir/parquet_file
    df = preporcess_create_hourly_borough_data(parquet_path, output_dir)
    all_df.append(df)

# Combine all sampled dataframes into one
combined_df = pd.concat(all_df, ignore_index=True, axis=0)
combined_df = (
    combined_df
    .sort_values(["Borough", "tpep_pickup_datetime"])
    .reset_index(drop=True)
)

combined_output_file = data_dir/"combined_sampled_data.csv"
combined_df.to_csv(combined_output_file, index=False)
logger.info(f"Combined sampled data saved to '{combined_output_file}' with shape: {combined_df.shape}")