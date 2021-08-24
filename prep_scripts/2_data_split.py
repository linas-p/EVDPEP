import pandas as pd
import numpy as np


DATA_PATH = "./data/EVconsumption/"


d1 = pd.read_csv(DATA_PATH + "data_1_selected.csv")
d1.head()


ids = np.unique(d1['trip_id'])
N = len(ids)
N_train = int(N * 0.7)
N_val = int(N * 0.8)
ids_train = ids[:N_train]
ids_val = ids[N_train:N_val]
ids_test = ids[N_val:]


data_train = d1[d1['trip_id'].isin(ids_train)]
data_val = d1[d1['trip_id'].isin(ids_val)]
data_test = d1[d1['trip_id'].isin(ids_test)]

data_train.shape, data_val.shape, data_test.shape


X_labels = ['speed', 'speed_limit', 'speed_avg_week', 'speed_avg_time', 'speed_avg_week_time', 'speed_avg', 'seconds', 'air_temperature',
       'wind_direction', 'wind_speed_ms', 'segangle',  
       'time', 'weekend',
       'drifting', 'dry', 'fog', 'freezing', 'none', 'snow', 'thunder', 'wet', 
        'living_street', 'motorway', 'motorway_link', 'primary', 'residential', 
       'secondary', 'secondary_link', 'service', 'tertiary', 
       'track', 'trunk', 'trunk_link', 'unclassified', 'unpaved']
y_labels = ['trip_id', 'trip_segmentno', 'segmentkey',
       'segmentid', 'ev_kwh']


data_train[X_labels].to_csv(DATA_PATH + "X_train.csv", index=False)
data_val[X_labels].to_csv(DATA_PATH + "X_val.csv", index=False)
data_test[X_labels].to_csv(DATA_PATH + "X_test.csv", index=False)       

data_train[y_labels].to_csv(DATA_PATH + "y_train.csv", index=False)
data_val[y_labels].to_csv(DATA_PATH + "y_val.csv", index=False)
data_test[y_labels].to_csv(DATA_PATH + "y_test.csv", index=False)