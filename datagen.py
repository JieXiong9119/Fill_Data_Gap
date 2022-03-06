import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### read site data, seperate by year
path_to_data = './data/'
datafilename = 'LF59_AMI_BCS.csv'
data = pd.read_csv(path_to_data+datafilename)
data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'])
data['Unnamed: 0'] = data['Unnamed: 0'].dt.strftime('%m-%d %H:%M')
data = data.rename({"Unnamed: 0": "TimeStr"}, axis=1)
data1 = data.iloc[0:17435]
data2 = data.iloc[32794:50692]
# print(data1)
# print(data2)

### TMY3 data
weafilename = 'Norfolk Naval epw.csv'
wea = pd.read_csv(path_to_data+weafilename)
wea['TIMESTAMP_x'] = wea['TIMESTAMP_x'].str.replace('24:00:00', '23:59:59')
wea['TIMESTAMP_x'] = pd.to_datetime(wea['TIMESTAMP_x'], format=' %m/%d  %H:%M:%S')
wea['TimeStr'] = wea['TIMESTAMP_x'].dt.strftime('%m-%d %H:%M')
wea['HOD'] = wea['TIMESTAMP_x'].dt.hour + wea['TIMESTAMP_x'].dt.minute / 60.0

### Combine site data with weather data
combinedata1 = pd.merge(wea, data1, how='left', on='TimeStr')
combinedata2 = pd.merge(wea, data2, how='left', on='TimeStr')

XY1 = combinedata1.dropna()
XY2 = combinedata2.dropna()
X1 = np.vstack((XY1.index.to_numpy(), XY1['HOD'].to_numpy(), XY1['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].to_numpy())).T
# X1 = np.vstack((XY1['HOD'].to_numpy(), XY1['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].to_numpy())).T
X2 = np.vstack((XY2.index.to_numpy(), XY2['HOD'].to_numpy(), XY2['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].to_numpy())).T

print([min(X1[:,0]), max(X1[:,0])])
print([min(X1[:,1]), max(X1[:,1])])
print([min(X1[:,2]), max(X1[:,2])])
X1[:,0] = X1[:,0] / 35040.0
X1[:,1] = X1[:,1] / 24.0
X1[:,2] = (X1[:,2]+20.0) / 60.0
print(X1.shape)
print([min(X1[:,0]), max(X1[:,0])])
print([min(X1[:,1]), max(X1[:,1])])
print([min(X1[:,2]), max(X1[:,2])])
plt.plot(X1[:,0]*35040.0, X1[:,2]*60.0-20.0, label = '2019')

X2[:,0] = X2[:,0] / 35040.0
X2[:,1] = X2[:,1] / 24.0
X2[:,2] = (X2[:,2]+20.0) / 60.0
print(X2.shape)
plt.plot(X2[:,0]*35040.0, X2[:,2]*60.0-20.0, label = '2020')
plt.show()

XY1.to_csv(path_to_data+'train_data_2019.csv')
XY2.to_csv(path_to_data+'train_data_2020.csv')

x = np.vstack((wea.index.to_numpy(), wea['HOD'].to_numpy(), wea['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].to_numpy())).T
print(x.shape)
np.savetxt(path_to_data+'test_x.csv', x, delimiter=",")
