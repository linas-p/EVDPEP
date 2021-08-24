import pandas as pd
import numpy as np

DATA_PATH = "./data/EVconsumption/"

weather = pd.read_csv(DATA_PATH + "dimweathermeasure.csv", sep = "|")
osm = pd.read_csv(DATA_PATH + "osm_dk_20140101.csv", sep = "|")
data0 = pd.read_csv(DATA_PATH + "2020_11_25_aal_viterbi.csv", sep = ",")
data1 = pd.read_csv(DATA_PATH + "2021_04_06_aal_north_viterbi.csv", sep = ",")
data2 = pd.read_csv(DATA_PATH + "2021_04_06_aal_south_viterbi.csv", sep = ",")

data = pd.concat([data0, data1, data2], axis=0)
data = data.drop_duplicates()


result = pd.merge(data, weather, how="left", on=["weathermeasurekey", "datekey"])
result = pd.merge(result, osm, how="left", on=["segmentkey"])

result.to_csv(DATA_PATH + "data_0_joined_data.csv")
print("Results {}".format(result.shape))