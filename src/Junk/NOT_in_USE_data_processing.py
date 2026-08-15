import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
import os
import logging
from pathlib import Path
import duckdb
# Setting up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

project_dir = Path().resolve()
data_dir = project_dir/"Data"

# logging configuration
logger = logging.getLogger('Data_Processing')
logger.setLevel(logging.DEBUG) 

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  

log_file_path = os.path.join(log_dir, 'Data_Processing.log') 
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)  

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("Beginning the data processing")

def load_data():
    try:
        con = duckdb.connect(database=':memory:', read_only=False)
        path=os.path.join(data_dir,"combined_sampled_data.parquet")
        df = con.execute(f"SELECT * FROM '{path}'").df()
        logger.info(f"Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def data_processing(data):
    try:
        logger.info(f"Beginning the data_processing function")
        loaction_id = pd.read_csv(data_dir/"location_ids.csv") ## Getting the names of the Locations ID's
        data = pd.merge(data, loaction_id, left_on="PULocationID",right_on="LocationID", how='left') ## merging loactions
        logger.info(f"data shape before processing {data.shape}")
        ##there are NAN values in Ratecode filling using below logic, the details were found on the google
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
        data.to_csv(os.path.join(data_dir,"preprocessed_data.csv"),index=False)
        return data
        
    except Exception as e:
        logger.error(f"Error while processing the data{e}")

data = load_data()
data= data_processing(data)

