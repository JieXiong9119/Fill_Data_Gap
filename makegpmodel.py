import sys
import csv
import numpy as np
import pandas as pd
import GPy
#import matplotlib.pyplot as plt
#import scipy.stats as stats
#from sklearn import preprocessing
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')
import pickle

with open('targets.csv') as f:
    reader = csv.reader(f)
    targets_all = next(reader)
    
print('Total '+str(len(targets_all))+' targets:')
print(targets_all)

def makemodel():

    path_to_data = './data/'
    targets = targets_all
    # targets = ['AVG_DEMAND_KW_CALCULATED_12','NSN_LF59_BLDG-CDW-LOOP_CDW-FLOW','NSN_LF59_BLDG-CDW-LOOP_CDW-T-R','NSN_LF59_BLDG-CDW-LOOP_CDW-T-S','NSN_LF59_COOLING-TOWER_FAN-HI','NSN_LF59_COOLING-TOWER_FAN-LO','NSN_LF59_COOLING-TOWER_FAN-STAT','NSN_LF59_CWP-1-VFD_VFD-SIG','NSN_LF59_CWP-1_PMP','NSN_LF59_CWP-1_PMP-STAT','NSN_LF59_BOILER-1_HW-T-E','NSN_LF59_BOILER-1_HW-T-L','NSN_LF59_PWSHP-2_CLG-COIL-T','NSN_LF59_PWSHP-2_OA-FLOW','NSN_LF59_PWSHP-2_SA-T','NSN_LF59_PWSHP-2_SA-T-STPT','NSN_LF59_WSHP-01_DA-T','NSN_LF59_WSHP-01_LW-T','NSN_LF59_WSHP-01_ZN-STPT-CL-EFF','NSN_LF59_WSHP-01_ZN-STPT-HT-EFF','NSN_LF59_WSHP-01_ZN-T','NSN_LF59_WSHP-12_DA-T','NSN_LF59_WSHP-12_LW-T','NSN_LF59_WSHP-12_ZN-STPT-CL-EFF','NSN_LF59_WSHP-12_ZN-STPT-HT-EFF','NSN_LF59_WSHP-12_ZN-T','NSN_LF59_BuildingKWhdSum','NSN_LF59_BuildingKWHdSumAnnualized']
    year = '2020'

    usage = "     Before using this script, you need to create a targets.csv file to map to all potential prediction target in data file\n\n" \
            "     Usage: python makegpmodel.py \n" \
            "        or: python makegpmodel.py -y <year> [<target_option>] <target>\n\n" \
            "         -y year of 2019, 2020\n" \
            "         -t target name in list\n" \
            "        -tn target number in list\n" \
            "   Example: python makegpmodel.py -y 2019 -t NSN_LF59_BLDG-CDW-LOOP_CDW-FLOW\n" \
            "        or: python makegpmodel.py -tn 3\n"

    try:
        if '-y' in sys.argv:
            year = sys.argv[sys.argv.index('-y') + 1]
        if '-t' in sys.argv:
            target_name = sys.argv[sys.argv.index('-t') + 1]
            if target_name in targets_all:
                targets = [target_name]
            elif target_name == 'all':
                targets = targets_all
            else:
                print('Target not found!')
                return
        if '-tn' in sys.argv:
            target_num = int(sys.argv[sys.argv.index('-tn') + 1])
            if target_num <= len(targets_all):
                targets = targets_all[target_num-1]
            else:
                print('Target not found!')
                return
    except Exception as e:
        print("ERROR - {}".format(e))
        print(usage)
        return
    
    datafilename = 'train_data_'+year+'.csv'
    data = pd.read_csv(path_to_data+datafilename)

    X = np.vstack((data.index.to_numpy(), data['HOD'].to_numpy(), data['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].to_numpy())).T
    X[:,0] = X[:,0] / 35040.0
    X[:,1] = X[:,1] / 24.0
    X[:,2] = (X[:,2]+20.0) / 60.0

    x_train = X

    ### train model
    for target in targets:
        print('Working on ' + target + '...')
        y_train = data[target].to_numpy()
        y_train = y_train[:, None]
        print(x_train.shape)
        print(y_train.shape)
        k = GPy.kern.RBF(x_train.shape[1], ARD=True) # The ARD = True is what makes GPy understand that there is
                                    # one lengthscale per dimension
        m = GPy.models.GPRegression(x_train, y_train, k)
        print(m)

        m.optimize_restarts( robust=True )
        print(m)
        print(m.rbf.lengthscale)

        with open('./model/'+year+'_'+target+'.pkl', 'wb') as file:
            pickle.dump(m, file)

        print('Model for ' + target + ' Complete!')

if __name__ == "__main__":
    makemodel()
