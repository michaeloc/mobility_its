import pandas as pd
import preprocess as prep
from tqdm import tqdm
import numpy as np
import gmplot.gmplot

stops_data = pd.read_csv('GTFSjaneiro/stops.txt')

data = pd.read_csv('data-500000.csv',delimiter=',', float_precision=None)

data['label'] = ['in_route' for i in range(data.shape[0])]

prepro = prep.PreProcess()

mapped_stops = dict()
for idx, row in tqdm(data.iterrows()):
    if row.velocidade < 5:
        for idx2,stop in stops_data.iterrows():
            dist = prepro.distance_in_meters([row.lat,row.lng], [stop.stop_lat,stop.stop_lon])
            if dist < 20:
                data.loc[idx,'label'] = 'bus_stop'

data.to_csv('data-500000-bus-stop-mapped.csv')