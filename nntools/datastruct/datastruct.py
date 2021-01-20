import re
import pandas as pd
import numpy as np
from scipy.stats import norm

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.latex.repr', True)

class DataStructure:
    '''
    Handle data tables and perform basic tidying tasks.
    '''

    def __init__(self, df=None, verbose=True, random_state=None, **kwargs):
        '''
        Initialise a data structure.

        Arguments:

            df:           the data structure to convert to DataFrame,
            verbose:      verbose output,
            random_state: set the random state,
            **kwargs:     additional arguments to pass to Pandas.
        '''

        self.random_state = random_state # random state
        self.path         = None         # path to the file
        self.features     = None         # list of feature names
        self.labels       = None         # list of label names

        self.df     = None # full dataset
        self.shape  = None # shape of the full dataset
        self.n_rows = None # no. of rows of the full dataset
        self.n_cols = None # no. of columns of the full dataset
        self.X      = None # input features in the full dataset
        self.y      = None # labels in the full dataset

        self.df_train     = None # training dataset
        self.shape_train  = None # shape of the training dataset
        self.n_rows_train = None # no. of rows of the training dataset
        self.X_train      = None # input features in the training dataset
        self.y_train      = None # labels in the training dataset

        self.df_validation     = None # validation dataset
        self.shape_validation  = None # shape of the validation dataset
        self.n_rows_validation = None # no. of rows of the validation dataset
        self.X_validation      = None # input features in the validation dataset
        self.y_validation      = None # labels in the validation dataset

        self.df_test     = None # test dataset
        self.shape_test  = None # shape of the test dataset
        self.n_rows_test = None # no. of rows of the test dataset
        self.X_test      = None # input features in the test dataset
        self.y_test      = None # labels in the test dataset

        # the data structure can be directly created in the constructor
        if df is not None:
            if verbose:
                print('\nCreating DataFrame directly.')

            # convert structure to dataframe
            self.df = pd.DataFrame(df, **kwargs)
            self.__update_rows_columns(split='full')

            if verbose:
                print(f'Shape of the dataset: {self.n_rows:d} rows, {self.n_cols:d} columns.')

    def __update_rows_columns(self, split='full'):
        '''
        Update the count of rows and columns.

        Arguments:
        
            split: one among ['full', 'train', 'validation', 'test'].
        '''

        if split == 'full':
            self.shape  = self.df.shape
            self.n_rows = self.shape[0]
            self.n_cols = self.shape[1]

        if split == 'train':
            if isinstance(self.df_train, pd.DataFrame):
                self.shape_train  = self.df_train.shape
                self.n_rows_train = self.shape_train[0]

            if isinstance(self.df_train, list):
                self.shape_train  = [df.shape for df in self.df_train]
                self.n_rows_train = [shape[0] for shape in self.shape_train]

        if split == 'validation':
            self.shape_validation  = self.df_validation.shape
            self.n_rows_validation = self.shape_validation[0]

        if split == 'test':
            self.shape_test  = self.df_test.shape
            self.n_rows_test = self.shape_test[0]

        return self

    def load_json(self, path, verbose=True, **kwargs):
        '''
        Load a JSON table.

        Arguments:

            path:     the path to the JSON file,
            verbose:  verbose output,
            **kwargs: additional arguments to pass to pandas.read_json.
        '''

        if verbose:
            print('\nLoading JSON dataset...')

        self.path   = path
        self.df     = pd.read_json(path, **kwargs)
        self.__update_rows_columns(split='full')

        if verbose:
            print(f'Shape of the dataset: {self.n_rows:d} rows, {self.n_cols:d} columns.')

        return self

    def load_csv(self, path, verbose=True, **kwargs):
        '''
        Load a CSV file.

        Arguments:

            path:     the path to the CSV file,
            verbose:  verbose output,
            **kwargs: additional arguments to pass to pandas.read_json.
        '''

        if verbose:
            print('\nLoading CSV dataset...')

        self.path   = path
        self.df     = pd.read_csv(path, **kwargs)
        self.__update_rows_columns(split='full')

        if verbose:
            print(f'Shape of the dataset: {self.n_rows:d} rows, {self.n_cols:d} columns.')

        return self

    def load_hdf(self, path, verbose=True, **kwargs):
        '''
        Load a HDF file.

        Arguments:

            path:     the path to the HDF file,
            verbose:  verbose output,
            **kwargs: additional arguments to pass to pandas.read_json.
        '''

        if verbose:
            print('\nLoading HDF dataset...')

        self.path   = path
        self.df     = pd.read_hdf(path, **kwargs)
        self.__update_rows_columns(split='full')

        if verbose:
            print(f'Shape of the dataset: {self.n_rows:d} rows, {self.n_cols:d} columns.')

        return self

    def load_table(self, path, verbose=True, **kwargs):
        '''
        Load a table based on the extension of the file.

        Arguments:

            path:     the path to the table,
            verbose:  verbose output,
            **kwargs: additional arguments to pass to pandas.
        '''

        if bool(re.search('json', path)):
            self.load_json(path, verbose=verbose, **kwargs)

        if bool(re.search('csv', path)):
            self.load_csv(path, verbose=verbose, **kwargs)

        if bool(re.search('h5', path)) or bool(re.search('hdf', path)):
            self.load_hdf(path, verbose=verbose, **kwargs)

        return self

    def describe(self, split='full', **kwargs):
        '''
        Print a description of the data.

        Arguments:

            split:    one among ['full', 'train', 'validation', 'test'],
            **kwargs: additional arguments for pd.describe().
        '''

        if split == 'full':
            return self.df.describe(**kwargs)

        if split == 'train' and self.df_train is not None:
            return self.df_train.describe(**kwargs)

        if split == 'validation' and self.df_validation is not None:
            return self.df_validation.describe(**kwargs)

        if split == 'test' and self.df_test is not None:
            return self.df_test.describe(**kwargs)

    def select(self, column, split='full'):
        '''
        Select column(s) of the dataframe.

        Argument:

            column: the name of the column(s) to select,
            split:  one of ['full', 'train', 'validation', 'test'].
        '''

        if split == 'full':
            return self.df[column]

        if split == 'train' and self.df_train is not None:
            return self.df_train[column]

        if split == 'validation' and self.df_validation is not None:
            return self.df_validation[column]

        if split == 'test' and self.df_test is not None:
            return self.df_test[column]

    def info(self, split='full'):
        '''
        Print infos on the data.

        Arguments:

            split: one among ['full', 'train', 'validation', 'test'].
        '''

        if split == 'full':
            self.df.info()

        if split == 'train' and self.df_train is not None:
            self.df_train.info()

        if split == 'validation' and self.df_validation is not None:
            self.df_validation.info()

        if split == 'test' and self.df_test is not None:
            self.df_test.info()

        return self

    def convert_dtypes(self, dtypes_dict, verbose=True):
        '''
        Convert dtypes in the dataframe.

        Arguments:

            dtypes_dict:  dict containing the map {column: new_dtype},
            verbose:      verbose output.
        '''

        assert isinstance(dtypes_dict, dict), 'Argument must be a dictionary!'

        # divide features and labels if necessary
        if self.features is not None:
            feat_dict = {feature: dtype for feature, dtype in dtypes_dict.items() if feature in self.features}
        if self.labels is not None:
            labs_dict = {label: dtype for label, dtype in dtypes_dict.items() if label in self.labels}

        # perform transformation
        self.df = self.df.astype(dtypes_dict)
        if self.X is not None:
            self.X = self.X.astype(feat_dict)
        if self.y is not None:
            self.y = self.y.astype(labs_dict)

        if self.df_train is not None:
            self.df_train = self.df_train.astype(dtypes_dict)

            if self.X_train is not None:
                self.X_train = self.X_train.astype(feat_dict)
            if self.y_train is not None:
                self.y_train = self.y_train.astype(labs_dict)

        if self.df_validation is not None:
            self.df_validation = self.df_validation.astype(dtypes_dict)

            if self.X_validation is not None:
                self.X_validation = self.X_validation.astype(feat_dict)
            if self.y_validation is not None:
                self.y_validation = self.y_validation.astype(labs_dict)

        if self.df_test is not None:
            self.df_test = self.df_test.astype(dtypes_dict)

            if self.X_test is not None:
                self.X_test = self.X_test.astype(feat_dict)
            if self.y_test is not None:
                self.y_test = self.y_test.astype(labs_dict)

        # verbose output
        if verbose:
            self.df.info()

        return self

    def replace_values(self, column, values_dict):
        '''
        Replace the values in a column using a dictionary.

        Arguments:

            column:      the name of the column,
            values_dict: the dictionary containing the translation.
        '''

        # replace in the datasets
        self.df.loc[:, column] = self.df[column].replace(values_dict)

        if self.features is not None and column in self.features:
            self.X[:, column] = self.X[column].replace(values_dict)
        if self.labels is not None and column in self.labels:
            self.y[:, column] = self.y[column].replace(values_dict)

        if self.df_train is not None:
            self.df_train.loc[:, column] = self.df_train[column].replace(values_dict)

            if self.features is not None and column in self.features:
                self.X_train[:, column] = self.X_train[column].replace(values_dict)
            if self.labels is not None and column in self.labels:
                self.y_train[:, column] = self.y_train[column].replace(values_dict)

        if self.df_validation is not None:
            self.df_validation.loc[:, column] = self.df_validation[column].replace(values_dict)

            if self.features is not None and column in self.features:
                self.X_validation[:, column] = self.X_validation[column].replace(values_dict)
            if self.labels is not None and column in self.labels:
                self.y_validation[:, column] = self.y_validation[column].replace(values_dict)

        if self.df_test is not None:
            self.df_test.loc[:, column] = self.df_test[column].replace(values_dict)

            if self.features is not None and column in self.features:
                self.X_test[:, column] = self.X_test[column].replace(values_dict)
            if self.labels is not None and column in self.labels:
                self.y_test[:, column] = self.y_test[column].replace(values_dict)

        return self

    def columns(self):
        '''
        Get the column names.
        '''

        return list(self.df.columns)

    def remove_columns(self, columns):
        '''
        Remove columns by name.
        
        Arguments:

            columns: list of columns to remove.
        '''

        self.df = self.df.drop(columns=columns)
        self.__update_rows_columns(split='full')

        if self.df_train is not None:
            self.df_train = self.df_train.drop(columns=columns)
            self.__update_rows_columns(split='train')

        if self.df_validation is not None:
            self.df_validation = self.df_validation.drop(columns=columns)
            self.__update_rows_columns(split='validation')

        if self.df_test is not None:
            self.df_test = self.df_test.drop(columns=columns)
            self.__update_rows_columns(split='test')

        return self

    def loc(self, id, split='full'):
        '''
        Localisation inside the data structure using boolean arrays.

        Arguments:

            id:    boolean array used to localise the data,
            split: one among ['full', 'train', 'validation', 'test'].
        '''

        if split == 'full':
            return self.df.loc[id]
        if split == 'train':
            return self.df_train.loc[id]
        if split == 'validation':
            return self.df_validation.loc[id]
        if split == 'test':
            return self.df_test.loc[id]

    def iloc(self, id, split='full'):
        '''
        Localisation inside the data structure using the index.

        Arguments:

            id:    list of index references to select,
            split: one among ['full', 'train', 'validation', 'test'].
        '''

        if split == 'full':
            return self.df.iloc[id]
        if split == 'train':
            return self.df_train.iloc[id]
        if split == 'validation':
            return self.df_validation.iloc[id]
        if split == 'test':
            return self.df_test.iloc[id]

    def train_test_split(self, test, train=None, validation=None, verbose=True):
        '''
        Split the data into training, validation and test folds.


        Arguments:
            
            test:       the size of the test set (in [0, 1]),
            train:      the size of the training set (in [0, 1]),
            validation: the size of the development set (in [0, 1]),
            verbose:    verbose output.
        '''

        # select the test set
        n_test       = int(test * self.n_rows)
        self.df_test = self.df.sample(n=n_test, random_state=self.random_state)
        self.__update_rows_columns(split='test')
        if verbose:
            print(f'Test set: {self.n_rows_test:d} rows ({100 * test:.2f}% ratio).')

        # remove the test from the full dataset
        df_oos = self.df.loc[~self.df.index.isin(self.df_test.index)]

        # select the validation set
        if validation is not None:
            n_validation       = int(validation * self.n_rows)
            self.df_validation = df_oos.sample(n=n_validation, random_state=self.random_state)
            self.__update_rows_columns(split='validation')
            if verbose:
                print(f'Validation set: {self.n_rows_validation:d} rows ({100 * validation:.2f}% ratio).')

            # remove the validation from the dataset
            df_oos = df_oos.loc[~df_oos.index.isin(self.df_validation.index)]

        # select the training set
        # if train is None then the remainder is the training set
        if train is None:
            self.df_train = df_oos
            self.__update_rows_columns(split='train')
            if verbose:
                if validation is not None:
                    print(f'Training set: {self.n_rows_train:d} rows ({100 * (1.0 - test - validation):.2f}% ratio).')
                else:
                    print(f'Training set: {self.n_rows_train:d} rows ({100 * (1.0 - test):.2f}% ratio).')
        else:
            n_train = int(train * self.n_rows)
            self.df_train = df_oos.sample(n=n_train, random_state=self.random_state)
            self.__update_rows_columns(split='train')
            if verbose:
                print(f'Training set: {self.n_rows_train:d} rows ({100 * train:.2f}% ratio).')

        # update features and labels if necessary
        if self.features is not None and self.labels is not None:
            self.create_features_labels(self.features, self.labels)

        return self

    def create_features_labels(self, features, labels):
        '''
        Separate features and labels.

        Arguments:

            features: list of column names forming the features,
            labels:   list of column names forming the labels.
        '''

        # sanitise the input
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
        if not isinstance(labels, list):
            self.labels = [labels]
        else:
            self.labels = labels

        # select features and labels
        self.X = self.df[self.features]
        self.y = self.df[self.labels]

        if self.df_train is not None:
            self.X_train = self.df_train[self.features]
            self.y_train = self.df_train[self.labels]

        if self.df_validation is not None:
            self.X_validation = self.df_validation[self.features]
            self.y_validation = self.df_validation[self.labels]

        if self.df_test is not None:
            self.X_test = self.df_test[self.features]
            self.y_test = self.df_test[self.labels]


        return self

    def normalise(self, low=0.0, high=1.0):
        '''
        Normalise the data in the interval [low, high] (using training data).
        
        Alias:
            normalize(low=0.0, high=1.0)

        Arguments:

            low:  the minimum of the rescaled distribution,
            high: the maximum of the rescaled distribution.
        '''
        
        # sanitise the input
        if low > high:
            low, high = high, low
        
        # define the scaling factor
        scale = np.abs(high - low)

        # get min and max and normalise
        df_min = self.X_train.min()
        df_max = self.X_train.max()

        self.X_train = scale * (self.X_train - df_min) / (df_max - df_min) + low

        self.X_test  = scale * (self.X_test - df_min) / (df_max - df_min) + low

        if self.X_validation is not None:
            self.X_validation = scale * ((self.X_validation - df_min) / (df_max - df_min) + low)

        return self

    def normalize(self, low=0.0, high=1.0):
        '''
        Normalise the data in the interval [low, high] (using training data).
        
        Alias:
            normalise(low=0.0, high=1.0)

        Arguments:

            low:  the minimum of the rescaled distribution,
            high: the maximum of the rescaled distribution.
        '''

        self.normalise(low=low, high=high)

    def standardise(self, with_mean=True):
        '''
        Standardise the data to have standard deviation 1 (using training data).

        Alias:
            standardize(with_mean=True)

        Arguments:

            with_mean: centre the values using the mean.
        '''
        
        mean = self.X_train.mean()
        std  = self.X_train.std()

        if not with_mean:
            mean = 0.0

        # standardise the data
        self.X_train = (self.X_train - mean) / std
        self.X_test  = (self.X_test - mean) / std

        if self.X_validation is not None:
            self.X_validation = (self.X_validation - mean) / std

        return self

    def standardize(self, with_mean=True):
        '''
        Standardize the data to have standard deviation 1 (using training data).

        Alias:
            standardise(with_mean=True)

        Arguments:

            with_mean: center the values using the mean.
        '''
        self.standardise(with_mean=with_mean)

    def label_rescaling(self, low=0.0, high=1.0):
        '''
        Rescale the labels in the interval [low, high] (using training labels).
        
        Arguments:

            low:  the minimum of the rescaled labels,
            high: the maximum of the rescaled labels.
        '''
        
        # sanitise the input
        if low > high:
            low, high = high, low
        
        # define the scaling factor
        scale = np.abs(high - low)

        # get min and max and normalise
        df_min = self.y_train.min()
        df_max = self.y_train.max()

        self.y_train = scale * (self.y_train - df_min) / (df_max - df_min) + low

        self.y_test  = scale * (self.y_test - df_min) / (df_max - df_min) + low

        if self.y_validation is not None:
            self.y_validation = scale * ((self.y_validation - df_min) / (df_max - df_min) + low)


        return self

    def apply(self, features=None, labels=None):
        '''
        Apply an arbitrary transformation to features and labels.

        Arguments:

            features: function (usually lambda) to apply to the features (can be a dict mapping column to function),
            labels:   function (usually lambda) to apply to the labels (can be a dict mapping column to function).
        '''
        
        if features is not None:
            if isinstance(features, dict):
                for feature, transformation in features.items():
                    self.X.loc[:, feature] = self.X[feature].apply(transformation)
                    
                    if self.X_train is not None:
                        self.X_train.loc[:, feature] = self.X_train[feature].apply(transformation)
                    if self.X_validation is not None:
                        self.X_validation.loc[:, feature] = self.X_validation[feature].apply(transformation)
                    if self.X_test is not None:
                        self.X_test.loc[:, feature] = self.X_test[feature].apply(transformation)
            else:
                self.X = self.X.apply(features)
                
                if self.X_train is not None:
                    self.X_train = self.X_train.apply(features)
                if self.X_validation is not None:
                    self.X_validation = self.X_validation.apply(features)
                if self.X_test is not None:
                    self.X_test = self.X_test.apply(features)

        if labels is not None:
            if isinstance(labels, dict):
                for label, transformation in labels.items():
                    self.y.loc[:, label] = self.y.apply(labels)
                    
                    if self.y_train is not None:
                        self.y_train.loc[:, label] = self.y_train[label].apply(transformation)
                    if self.y_validation is not None:
                        self.y_validation.loc[:, label] = self.y_validation[label].apply(transformation)
                    if self.y_test is not None:
                        self.y_test.loc[:, label] = self.y_test[label].apply(transformation)
            else:
                self.y = self.y.apply(labels)
                
                if self.y_train is not None:
                    self.y_train = self.y_train.apply(labels)
                if self.y_validation is not None:
                    self.y_validation = self.y_validation.apply(labels)
                if self.y_test is not None:
                    self.y_test = self.y_test.apply(labels)

        return self

    def __compute_iqr(self, feature, standardise=False):
        '''
        Compute the whiskers of a variable using training data.

        Arguments:
            
            feature:     the variable to study,
            standardise: whether to standardise the feature or not.
        '''

        if standardise:
            variable = (self.X_train[feature] - self.X_train[feature].mean()) / self.X_train[feature].std()
        else:
            variable = self.X_train[feature]

        quartile = np.quantile(variable, [0.25, 0.75])
        iqr      = quartile[1] - quartile[0]
        whiskers = (quartile[0] - 1.5 * iqr, quartile[1] + 1.5 * iqr)

        return quartile, whiskers

    def test_distribution(self, verbose=False):
        '''
        Test the distribution of a variable using training data.

        Arguments:

            verbose: verbose output.
        '''

        # center and compare
        gaussianity = {}
        for feature in self.X_train:
            quartile, _ = self.__compute_iqr(feature, standardise=True)
            gauss    = norm.ppf([0.25, 0.75])
            diff     = np.abs(quartile - gauss).min()
            gaussianity[feature] = diff

            if verbose:
                print(f'Empirical quartiles in {feature} (standardised) differ at most {diff:f} from a normal distribution.')

        return gaussianity

    def filter_outliers(self, methods):
        '''
        Filter the outliers using training data.

        Arguments:

            method: dict containing the method for each column (either tuple/list containing the boundaries or 'iqr' for interquartile)
        '''
        
        outliers = {}
        for feature, method in methods.items():

            if method == 'iqr':
                _, whiskers = self.__compute_iqr(feature)
                outliers[feature] = whiskers

            else:
                outliers[feature] = method

        # remove outliers
        for feature, out in outliers.items():
            self.y_train = self.y_train.loc[(self.X_train[feature] >= out[0]) &
                                            (self.X_train[feature] <= out[1])
                                           ]
            self.X_train = self.X_train.loc[(self.X_train[feature] >= out[0]) &
                                            (self.X_train[feature] <= out[1])
                                           ]

            self.df_train = pd.concat([self.X_train, self.y_train], axis=1)
        self.__update_rows_columns(split='train')

    def compute_bins(self, column, bins=None, replace=False):
        '''
        Compute binning for a variable.

        Arguments:

            column:  the variable to bin,
            bins:    a list of bins or a number of bins,
            replace: add a column (True) or replace the existing column.
        '''
        
        # sanitise the input
        if bins is not None:
            if not isinstance(bins, int):
                bins = list(bins)
        else:
            bins = 10

        if isinstance(bins, list):
            if bins[0] > self.df[column].min():
                bins = [self.df[column].min()] + bins
            if bins[1] < self.df[column].max():
                bins = bins + [self.df[column].max()]

        # compute the bins
        if replace:
            self.df[:, column] = pd.cut(self.df[column], bins=bins, right=False, include_lowest=True)

            if self.df_train is not None:
                self.df_train[:, column] = pd.cut(self.df_train[column], bins=bins, right=False, include_lowest=True)
            if self.df_validation is not None:
                self.df_validation[:, column] = pd.cut(self.df_validation[column], bins=bins, right=False, include_lowest=True)
            if self.df_test is not None:
                self.df_test[:, column] = pd.cut(self.df_test[column], bins=bins, right=False, include_lowest=True)
        else:
            self.df[column + '_bins'] = pd.cut(self.df[column], bins=bins, right=False, include_lowest=True)

            if self.df_train is not None:
                self.df_train[column + '_bins'] = pd.cut(self.df_train[column], bins=bins, right=False, include_lowest=True)
            if self.df_validation is not None:
                self.df_validation[column + '_bins'] = pd.cut(self.df_validation[column], bins=bins, right=False, include_lowest=True)
            if self.df_test is not None:
                self.df_test[column + '_bins'] = pd.cut(self.df_test[column], bins=bins, right=False, include_lowest=True)

        # update
        if self.features is not None and self.labels is not None:

            if not replace:
                if column in self.features:
                    self.features.append(column + '_bins')
                if column in self.labels:
                    self.labels.append(column + '_bins')

            self.create_features_labels(self.features, self.labels)

        return self
