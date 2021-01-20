import unittest
from nntools import *


# create dummy data
n_rows, n_cols = 5000, 15

dummy = {'column_' + str(col): list(np.random.normal(loc=2.0, scale=1.5, size=(n_rows,))) for col in range(n_cols)}
dummy['label'] = list(np.random.choice([0, 1], size=(n_rows,)))

# create test files
df = pd.DataFrame(dummy)
df.to_json('./test.json.gz')
df.to_csv('./test.csv.gz', index=False)
df.to_hdf('./test.h5', 'test')

# test class
class TestDataStructure(unittest.TestCase):

    def test_constructor(self):

        print('\nTesting constructor...')

        data = DataStructure(dummy)

        assert data.shape  == (n_rows, n_cols + 1)
        assert data.n_rows == n_rows
        assert data.n_cols == n_cols + 1

    def test_load_json(self):

        print('\nTesting JSON loader...')

        data = DataStructure()
        data = data.load_json('./test.json.gz', orient='columns')

        assert data.shape  == (n_rows, n_cols + 1)
        assert data.n_rows == n_rows
        assert data.n_cols == n_cols + 1

    def test_load_csv(self):

        print('\nTesting CSV loader...')

        data = DataStructure()
        data = data.load_csv('./test.csv.gz')

        assert data.shape  == (n_rows, n_cols + 1)
        assert data.n_rows == n_rows
        assert data.n_cols == n_cols + 1

    def test_load_hdf(self):

        print('\nTesting HDF loader...')

        data = DataStructure()
        data = data.load_hdf('./test.h5')

        assert data.shape  == (n_rows, n_cols + 1)
        assert data.n_rows == n_rows
        assert data.n_cols == n_cols + 1

    def test_load_table(self):

        print('\nTesting generic loader...')

        data = DataStructure()
        data = data.load_table('./test.json.gz')

        assert data.shape  == (n_rows, n_cols + 1)
        assert data.n_rows == n_rows
        assert data.n_cols == n_cols + 1

        data = DataStructure()
        data = data.load_table('./test.csv.gz')

        assert data.shape  == (n_rows, n_cols + 1)
        assert data.n_rows == n_rows
        assert data.n_cols == n_cols + 1

        data = DataStructure()
        data = data.load_table('./test.h5')

        assert data.shape  == (n_rows, n_cols + 1)
        assert data.n_rows == n_rows
        assert data.n_cols == n_cols + 1

    def test_convert_dtypes(self):

        print('\nTesting dtype conversion...')

        data = DataStructure(dummy)

        data.train_test_split(train=0.8, validation=0.1, test=0.1)

        features = ['column_1', 'column_2']
        labels   = 'label'
        data.create_features_labels(features, labels)

        data.convert_dtypes({'column_1': np.int, 'label': np.int})

    def test_replace_values(self):

        print('\nTesting value replacement...')

        data = DataStructure(dummy)

        replacement = {0: 'A', 1: 'B'}

        data.replace_values('label', replacement)

        assert ('A' in pd.unique(data.select('label'))) and ('B' in pd.unique(data.select('label')))

    def test_describe(self):

        print('\nTesting description method...')

        data = DataStructure(dummy)
        assert isinstance(data.describe(), pd.DataFrame)

    def test_remove_columns(self):

        print('\nTesting column remover...')
        
        data = DataStructure(dummy)
        data.remove_columns(['column_1', 'column_2'])

        assert data.shape  == (n_rows, n_cols - 1)
        assert data.n_rows == n_rows
        assert data.n_cols == n_cols - 1
        assert 'column_1' not in data.columns()
        assert 'column_2' not in data.columns()

    def test_train_test_split(self):

        print('\nTesting train:validation:test splitter...')

        splits = {'train':      0.7,
                  'validation': 0.2,
                  'test':       0.1
                 }

        data = DataStructure(dummy)
        data.train_test_split(**splits)

        assert data.n_rows_train      == int(splits['train'] * data.n_rows)
        assert data.n_rows_validation == int(splits['validation'] * data.n_rows)
        assert data.n_rows_test       == int(splits['test'] * data.n_rows)

        splits = {'test': 0.1}

        data = DataStructure(dummy)
        data.train_test_split(**splits)

        assert data.n_rows_train == int((1.0 - splits['test']) * data.n_rows)
        assert data.n_rows_test  == int(splits['test'] * data.n_rows)

    def test_create_features_labels(self):

        print('\nTesting feature/label splitter...')

        features = ['column_1', 'column_2']
        labels   = 'label'

        splits = {'test': 0.1}

        data = DataStructure()
        data.load_table('./test.csv.gz')
        data.train_test_split(**splits)

        data.create_features_labels(features, labels)

        assert data.X_train.shape[1] == len(features)
        assert data.y_train.shape[1] == 1
        assert data.X_test.shape[1] == len(features)
        assert data.y_test.shape[1] == 1

    def test_normalise(self):

        print('\nTesting normalisation methods...')

        features = ['column_1', 'column_2']
        labels   = 'label'

        # check with single split
        splits = {'train':      0.7,
                  'validation': 0.2,
                  'test':       0.1
                 }

        data = DataStructure()
        data.load_table('./test.h5')
        data.train_test_split(**splits)
        data.create_features_labels(features, labels)

        # test usual case
        low, high = -1.0, 1.0
        data.normalise(low=low, high=high)

        assert data.X_train.max().max() == high
        assert data.X_train.min().min() == low

        # test inverted (wrong) case
        low, high = 1.0, -3.0
        data.normalize(low=low, high=high)

        assert data.X_train.max().max() == low
        assert data.X_train.min().min() == high

    def test_label_rescaling(self):

        print('\nTesting label rescaling...')

        features = ['column_1', 'column_2']
        labels   = 'label'

        # check with single split
        splits = {'train':      0.7,
                  'validation': 0.2,
                  'test':       0.1
                 }

        data = DataStructure()
        data.load_table('./test.h5')
        data.train_test_split(**splits)
        data.create_features_labels(features, labels)

        # test usual case
        low, high = -1.0, 1.0
        data.label_rescaling(low=low, high=high)

        assert data.y_train[labels].max() == high
        assert data.y_train[labels].min() == low

    def test_standardisation(self):

        print('\nTesting standardisation methods...')

        features = ['column_1', 'column_2']
        labels   = 'label'

        # check with single split
        splits = {'train':      0.7,
                  'validation': 0.2,
                  'test':       0.1
                 }

        # check the case with centering
        data = DataStructure()
        data.load_table('./test.json.gz')
        data.train_test_split(**splits)
        data.create_features_labels(features, labels)

        # test usual case
        data.standardize(with_mean=True)

        for f in features:
            assert np.isclose(data.X_train.mean()[f], 0.0)
            assert np.isclose(data.X_train.std()[f], 1.0)
        
        # check the case without centering
        data = DataStructure()
        data.load_table('./test.json.gz')
        data.train_test_split(**splits)
        data.create_features_labels(features, labels)

        # test usual case
        data.standardise(with_mean=False)

        for f in features:
            assert np.abs(data.X_train[f].mean()) >= 0.0
            assert np.isclose(data.X_train[f].std(), 1.0)

    def test_apply(self):

        print('\nTesting application of functions...')

        features = ['column_1', 'column_2']
        labels   = 'label'

        # check with single split
        splits = {'train':      0.7,
                  'validation': 0.2,
                  'test':       0.1
                 }

        data = DataStructure()
        data.load_table('./test.json.gz')
        data.train_test_split(**splits)
        data.create_features_labels(features, labels)

        # test usual case
        data.normalise(low=0.0, high=1.0)

        # apply exponential
        data.apply(features={'column_1': np.exp, 'column_2': np.log1p},
                   labels=lambda x: x + 1
                  )

        assert data.X_train['column_1'].min() == 1.0
        assert data.X_train['column_2'].max() == np.log(2)
        assert data.y_train['label'].max() == 2

    def test_test_distribution(self):

        print('\nTesting distribution of the features...')

        features = ['column_1', 'column_2', 'column_3', 'column_4']
        labels   = 'label'

        # check with single split
        splits = {'train':      0.7,
                  'validation': 0.2,
                  'test':       0.1
                 }

        data = DataStructure()
        data.load_table('./test.json.gz')
        data.train_test_split(**splits)
        data.create_features_labels(features, labels)

        gauss = data.test_distribution(verbose=True)
        gauss = np.mean(list(gauss.values()))

        assert gauss < 1.0

    def test_filter_outliers(self):

        print('\nTesting outliers filter...')

        features = ['column_1', 'column_2']
        labels   = 'label'

        # check with single split
        splits = {'train':      0.7,
                  'validation': 0.2,
                  'test':       0.1
                 }

        data = DataStructure()
        data.load_table('./test.json.gz')
        data.train_test_split(**splits)
        data.create_features_labels(features, labels)

        n_rows = data.n_rows_train

        methods = {'column_1': (1.9, 2.1),
                   'column_2': 'iqr'
                  }

        data.filter_outliers(methods)
        
        assert np.all(data.X_train.index == data.y_train.index)
        assert data.n_rows_train < n_rows

    def test_compute_bins(self):

        print('\nTesting bin computations...')

        features = ['column_1', 'column_2']
        labels   = 'label'

        splits = {'train': 0.9, 'test': 0.1}

        data = DataStructure(dummy)
        data.train_test_split(**splits)
        data.create_features_labels(features, labels)

        data.compute_bins('label', bins=3, replace=False)

        assert 'label_bins' in data.columns()
        assert 'label_bins' not in data.X.columns
        assert 'label_bins' in data.y.columns
        assert 'label_bins' not in data.X_train.columns
        assert 'label_bins' in data.y_train.columns
        assert 'label_bins' not in data.X_test.columns
        assert 'label_bins' in data.y_test.columns
