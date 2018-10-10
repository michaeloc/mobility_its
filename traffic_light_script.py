import pandas as pd
import preprocess as prep
from tqdm import tqdm
import numpy as np
import gmplot.gmplot

traffic_stops_data = pd.read_csv('semaforos.csv', sep=';')

data = pd.read_csv('data-100000-bus-stop-mapped.csv',delimiter=',', float_precision=None)

# data['label'] = ['in_route' for i in range(data.shape[0])]

prepro = prep.PreProcess()

for idx, row in tqdm(data.iterrows()):
    if row.velocidade < 5:
        for idx2,stop in traffic_stops_data.iterrows():
            dist = prepro.distance_in_meters([row.lat,row.lng], [stop.Latitude,stop.Longitude])
            if dist < 30 and row.label != 'bus_stop':
                data.loc[idx,'label'] = 'traffic_light'

data.to_csv('data-100000-traffic-light.csv')