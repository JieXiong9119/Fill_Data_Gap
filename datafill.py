import numpy as np
import pandas as pd
import GPy
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import preprocessing
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')
import pickle

with open('save.pkl', 'rb') as file:
    m = pickle.load(file)

# path_to_data = './data/'
# traindatafilename = 'train_data_2019.csv'
# data = pd.read_csv(path_to_data+traindatafilename)

# X = np.vstack((data.index.to_numpy(), data['HOD'].to_numpy(), data['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].to_numpy())).T
# X[:,0] = X[:,0] / 35040.0
# X[:,1] = X[:,1] / 24.0
# X[:,2] = (X[:,2]+20.0) / 60.0

path_to_data = './data/'
testdatafilename = 'test_x.csv'
x_test = np.loadtxt(path_to_data+testdatafilename, delimiter=',')

x_test[:,0] = x_test[:,0] / 35040.0
x_test[:,1] = x_test[:,1] / 24.0
x_test[:,2] = (x_test[:,2]+20.0) / 60.0


