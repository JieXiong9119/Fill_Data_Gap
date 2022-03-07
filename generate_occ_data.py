import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

path_to_data='./pred/'

dts = [dt.strftime('%Y-%m-%d %H:%M') for dt in 
       datetime_range(datetime(2019, 1, 1, 0, 15), datetime(2020, 12, 31, 23, 45), 
       timedelta(minutes=15))]

#print(dts)

df = pd.DataFrame({'Timestamp': dts})
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M')

# Monday=0, Sunday=6
df['dayofweek'] = df['Timestamp'].dt.dayofweek
df['hour'] = df['Timestamp'].dt.hour
df['occupied'] = 0

df.loc[(df['dayofweek']<5) & (df['hour']>7) & (df['hour']<18), 'occupied'] = 1
df.to_csv(path_to_data+'occ.csv')
