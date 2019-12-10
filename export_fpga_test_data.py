#%%
# import os
import numpy as np
import sys
sys.path.append("..")
from soil_classifier.dataset import Landsat
from sklearn.preprocessing import OneHotEncoder 

SEED = 0
np.random.seed(SEED)

# cwd = os.getcwd()
DATA_FOLDER = '../data/'
DATASET_NAME = 'Landsat'

#%%

def main():
    dataset = Landsat(data_folder=DATA_FOLDER)

    x_train, y_train, x_test, y_test = dataset.load(shuffle=True, seed=SEED)
    num_classes = dataset.num_classes
    num_bands = dataset.num_bands

    # dataset standarization
    x_mean = np.mean(x_train,axis=0)
    x_std = np.std(x_train,axis=0)

    x_mean = np.transpose(x_mean[:,np.newaxis])
    x_std = np.transpose(x_std[:,np.newaxis])

    # Replace zero sigma values with 1
    x_std[x_std == 0] = 1

    x_train_norm = np.divide( (x_train-x_mean), x_std)
    x_test_norm = np.divide( (x_test-x_mean), x_std)

    # labels to one hot encoding
    onehotencoder = OneHotEncoder(categories='auto') 
    y_train = onehotencoder.fit_transform(y_train[:,np.newaxis]).toarray()
    y_test = onehotencoder.fit_transform(y_test[:,np.newaxis]).toarray() 

    print('\nSaving data in text (dat) file ...')
    np.savetxt(DATA_FOLDER+DATASET_NAME+'_x_train.dat', x_train_norm, fmt='%.6f')
    np.savetxt(DATA_FOLDER+DATASET_NAME+'_y_train.dat', y_train, fmt='%.6f')
    np.savetxt(DATA_FOLDER+DATASET_NAME+'_x_test.dat', x_test_norm, fmt='%.6f')
    np.savetxt(DATA_FOLDER+DATASET_NAME+'_y_test.dat', y_test, fmt='%.6f')
    print('done!')

if __name__ == "__main__":
    main()