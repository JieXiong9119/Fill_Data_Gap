#import arviz as az
#import pickle
import numpy as np
import pandas as pd
import pymc3 as pm
import theano
#import theano.tensor as T
#from sklearn import datasets
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

__all__ = ['construct_nn', 'generate_data']

def construct_nn(X, Y, tardis='Normal', n_hidden=5):

    floatX = theano.config.floatX
    RANDOM_SEED = 12356
    rng = np.random.default_rng(RANDOM_SEED)

    # Initialize random weights between each layer
    init_1 = rng.standard_normal(size=(X.shape[1], n_hidden)).astype(floatX)
    init_2 = rng.standard_normal(size=(n_hidden, n_hidden)).astype(floatX)
    init_out = rng.standard_normal(size=n_hidden).astype(floatX)

    coords = {
        "hidden_layer_1": np.arange(n_hidden),
        "hidden_layer_2": np.arange(n_hidden),
        "train_cols": np.arange(X.shape[1]),
        "obs_id": np.arange(X.shape[0]),
    }
    with pm.Model(coords=coords) as neural_network:

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal(
            "w_in_1", 0, sigma=1, testval=init_1, dims=("train_cols", "hidden_layer_1")
        )

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal(
            "w_1_2", 0, sigma=1, testval=init_2, dims=("hidden_layer_1", "hidden_layer_2")
        )

        # Weights from hidden layer to output
        weights_2_out = pm.Normal("w_2_out", 0, sigma=1, testval=init_out, dims="hidden_layer_2")

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(X, weights_in_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
        act_out = pm.math.sigmoid(pm.math.dot(act_2, weights_2_out))

        # 'Normal' for continuous target, 'Bernoulli' for binary target
        if tardis == 'Normal':
            out = pm.Normal(
                "out",
                mu=act_out,
                observed=Y,
            )
        elif tardis == 'Bernoulli':
            out = pm.Bernoulli(
                "out",
                act_out,
                observed=Y,
                #total_size=X_train.shape[0],  # IMPORTANT for minibatches
                #dims="obs_id",
            )

    return neural_network

def train_model(target, x_train, y_train, tardis, n_hidden=5, save_to_db=True):
    
    print('Working on ' + target + '...')

    neural_network = construct_nn(X=x_train, Y=y_train, tardis=tardis, n_hidden=n_hidden)
    pm.set_tt_rng(42)
    with neural_network:
        inference = pm.ADVI()
        approx = pm.fit(n=3000, method=inference)

    trace = approx.sample(draws=1000)
    if save_to_db == True:
        pm.save_trace(trace=trace, directory= './model/'+target)

    print('Model for ' + target + ' Complete!')

    return trace

def generate_data(path_to_data='./data/', datafilename='LF59_AMI_BCS.csv', weafilename='Norfolk Naval epw.csv'):
    
    ### read site data, seperate by year
    data = pd.read_csv(path_to_data+datafilename)
    data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'])
    data['Unnamed: 0'] = data['Unnamed: 0'].dt.strftime('%m-%d %H:%M')
    data = data.rename({"Unnamed: 0": "TimeStr"}, axis=1)
    data1 = data.iloc[0:17435]
    data2 = data.iloc[32794:50692]
    # print(data1)
    # print(data2)

    ### TMY3 data
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
