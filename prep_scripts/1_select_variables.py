import pandas as pd
import numpy as np
import datetime
from utils import *
import matplotlib.pyplot as plt


DATA_PATH = "./data/EVconsumption/"


d1 = pd.read_csv(DATA_PATH + "data_0_joined_data.csv")


data = d1[['trip_id', 'trip_segmentno', 'segmentkey', 'segmentid', 'datekey', 'direction_x', 
           'timekey', 'speed', 'meters_driven', 'meters_segment', 
           'seconds', 'class', 'air_temperature', 'wind_direction', 
           'wind_speed_ms', 'category', 'segangle',
           'speedlimit_forward', 'speedlimit_backward', 'ev_kwh']]

speed_limits = data[data['speedlimit_forward'] != 0].groupby('category').agg({'speedlimit_forward':['median']})

data['weekend'] = to_weekend(data['datekey'])


data['speed_limit'] = data['speedlimit_forward']
data['speed_limit'][data['direction_x'] == "BACKWARD"] = data['speedlimit_backward'][data['direction_x'] == "BACKWARD"]
names = speed_limits.index
limits = speed_limits.to_numpy()

for j in range(data.shape[0]):
    if data['speed_limit'][j] == 0:
        """ """
        idx = np.where(data.loc[j, 'category'] == names)
        if len(idx[0]) > 0:
            lim = limits[idx]
        else: ## any other case
            lim = 30
        data['speed_limit'][j] = lim


speed_limits_description = data[data['speedlimit_forward'] != 0].groupby('category').agg(
    {'speedlimit_forward':['min','median','max']})
print(speed_limits_description.to_latex())


da = pd.crosstab(data['category'], data['speed_limit'])
print(da.to_latex())



#plt.grid()
plt.hist(data['timekey'][data['weekend'] == 0], bins=24, alpha=0.5, label='Workday', density=False)
plt.hist(data['timekey'][data['weekend'] == 1], bins=24, alpha=0.5, label='Weekend', density=False)
plt.legend(loc='upper right')
plt.xlabel("Day time (h)")
ticks_at = np.array([0, 700, 1100, 1600, 2400])
plt.xticks(ticks_at, (ticks_at // 100).astype(np.int), rotation=0)
plt.ylabel("Measuments count")
plt.savefig("intensity.pdf", dpi = 600)        


data['time'] = 1
data['time'][data['timekey'] < 600] = 0
data['time'][data['timekey'] > 2200] = 0
data['time'][np.logical_and(data['timekey'] > 700, data['timekey'] < 900)] = 2
data['time'][np.logical_and(data['timekey'] > 1500, data['timekey'] < 1700)] = 2  

data = pd.concat([data, pd.get_dummies(data['class'])], axis=1)
data = pd.concat([data, pd.get_dummies(data['category'])], axis=1)

d1 = data.drop(['datekey', 'direction_x', 'timekey', 'class', 'category', 
               'speedlimit_forward','meters_segment',  'speedlimit_backward'], axis=1)

d2 = d1[d1['ev_kwh'].notna()]
d2 = d2[d2['wind_direction'].notna()]


descriptive = d2[['speed', 'seconds','meters_driven', 'air_temperature', 
        'wind_speed_ms', 'segangle', 'ev_kwh',]].describe()
descriptive = descriptive.round(2)
print(descriptive.to_latex(index=True)) 





d2['speed_avg_week'] = 0
for seg in np.unique(d2['segmentkey']):
    for val in np.unique(d2['weekend']):
        idx = d2['segmentkey'] == seg
        idx_w = d2['weekend'] == val
        ids = idx & idx_w
        speed = np.mean(d2[ids]['speed'])
        d2.loc[ids, 'speed_avg_week'] = speed

d2['speed_avg_time'] = 0
for seg in np.unique(d2['segmentkey']):
    for tim in np.unique(d2['time']):
        idx = d2['segmentkey'] == seg
        idx_w = d2['time'] == tim
        ids = idx & idx_w
        speed = np.mean(d2[ids]['speed'])
        d2.loc[ids,'speed_avg_time'] = speed

            
d2['speed_avg_week_time'] = 0
for seg in np.unique(d2['segmentkey']):
    for tim in np.unique(d2['time']):
        for val in np.unique(d2['weekend']):
            idx = d2['segmentkey'] == seg
            idx_w = d2['weekend'] == val
            idx_t = d2['time'] == tim
            ids = idx & idx_w & idx_t
            speed = np.mean(d2[ids]['speed'])
            d2.loc[ids, 'speed_avg_week_time'] = speed

d2['speed_avg'] = 0
for seg in np.unique(d2['segmentkey']):
        idx = d2['segmentkey'] == seg
        ids = idx 
        speed = np.mean(d2[ids]['speed'])
        d2.loc[ids, 'speed_avg'] = speed

d2.to_csv(DATA_PATH + "data_1_selected.csv")           