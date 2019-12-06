#%%
import os
import requests
import numpy as np 
from sklearn.preprocessing import OneHotEncoder 

# https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)

#%%
DATASET_FILES = ['sat.doc', 'sat.trn', 'sat.tst']

class Landsat():
    def __init__(self, data_folder=None):
        cwd = os.getcwd()

        self.name = 'Landsat'
        self.url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/'

        if data_folder is None:
            self.data_folder = cwd+'/data'
        else:
            self.data_folder = data_folder

        self.train_file = self.data_folder + '/' + DATASET_FILES[1]
        self.test_file = self.data_folder + '/' + DATASET_FILES[2]

        self.classes_names=(
                        'red soil',
                        'cotton crop',
                        'grey soil',
                        'damp grey soil',
                        'soil veg stubble',
                        'very damp grey soil')
        self.num_classes = len(self.classes_names)
        self.num_bands = 4
        self.data_dim = 36

        self.x_train = []
        self.y_train = []

        self.x_test = []
        self.y_test = []

        self.x_mean = None
        self.x_std = None
        self.x_max = None

        self.shuffle = True
        self.seed = None
    
    def _check_host(self):
        check = False
        try:
            f = open(self.train_file, 'r')
            f.close()
            f = open(self.test_file, 'r')
            f.close()
            check = True
        except:
            pass

        return check

    def _download(self):
        for f in DATASET_FILES:
            dfile = requests.get(self.url + f)
            open(self.data_folder + '/' + f, 'wb').write(dfile.content)
        return

    def _process(self, shuffle=True, seed=None):
        np.random.seed(seed)

        # read csv files (originals)
        try:
            train_data = np.genfromtxt(self.train_file, delimiter=' ')
            test_data = np.genfromtxt(self.test_file, delimiter=' ')
        except:
            raise IOError('Dataset original files have not found!')

        # convert to train/test arrays
        self.x_train = train_data[:,:-1]
        self.y_train = train_data[:,-1]
        self.x_test = test_data[:,:-1]
        self.y_test = test_data[:,-1]

        # replace class 7 to 6, because class 6 has no data
        train_count = np.count_nonzero(self.y_train == 6)
        test_count = np.count_nonzero(self.y_test == 6)

        if train_count == 0 and test_count == 0:
            self.y_train[self.y_train == 7] = 6
            self.y_test[self.y_test == 7] = 6
        
        # shuffling
        if shuffle:
            train_idx = np.random.permutation(self.x_train.shape[0])
            test_idx = np.random.permutation(self.x_test.shape[0])

            self.x_train = self.x_train[train_idx]
            self.y_train = self.y_train[train_idx]
            self.x_test = self.x_test[test_idx]
            self.y_test = self.y_test[test_idx]


        # misc data
        self.data_dim = self.x_train.shape[1]
        self.num_classes = np.max(self.y_train)
        self.shuffle = shuffle
        self.seed = seed
        return


    def load(self, shuffle=True, seed=None):
        if not self._check_host():
            print('Downloading dataset from {}'.format(self.url))
            self._download()
            self._process(shuffle=shuffle, seed=seed)
            if not self._check_host():
                raise IOError('Download error!')
        else:
            self._process(shuffle=shuffle, seed=seed)

        return self.x_train, self.y_train, self.x_test, self.y_test

    def posprocess(self, x_proc_type='standarization', y_proc_type='one-hot'):
        self._process(shuffle=self.shuffle, seed=self.seed)

        if x_proc_type == 'standarization':
            # dataset standarization
            x_mean = np.mean(self.x_train,axis=0)
            x_std = np.std(self.x_train,axis=0)

            x_mean = np.transpose(x_mean[:,np.newaxis])
            x_std = np.transpose(x_std[:,np.newaxis])

            # Replace zero sigma values with 1
            x_std[x_std == 0] = 1

            self.x_train = np.divide( (self.x_train-x_mean), x_std)
            self.x_test = np.divide( (self.x_test-x_mean), x_std)
            self.x_mean = x_mean
            self.x_std = x_std
        elif x_proc_type == 'normalization':
            x_max = np.max(self.x_train,axis=0)
            x_max[x_max == 0] = 1

            self.x_train = np.divide(self.x_train, x_max)
            self.x_test = np.divide(self.x_test, x_max)     

            self.x_max = x_max

        if y_proc_type == 'one-hot':
            # labels to one hot encoding
            onehotencoder = OneHotEncoder(categories='auto') 
            self.y_train = onehotencoder.fit_transform(self.y_train[:,np.newaxis]).toarray()
            self.y_test = onehotencoder.fit_transform(self.y_test[:,np.newaxis]).toarray() 

        return self.x_train, self.y_train, self.x_test, self.y_test
            
