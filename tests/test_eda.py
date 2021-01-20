import unittest
from nntools import *

# load dummy data
data = DataStructure()
data = data.load_table('./test.csv.gz')
data = data.train_test_split(test=0.1, validation=0.1)
data = data.create_features_labels(['column_1', 'column_2', 'column_3', 'column_4'], ['label'])

class TestEDA(unittest.TestCase):

    def test_constructor(self):

        print('\nTesting constructor method...')

        eda = EDA(data)

    def test_compute_corr(self):

        print('\nTesting correlation...')

        eda = EDA(data)
        eda.compute_corr(split='full', regex='^column_[1-3]$')

        assert eda.corr is not None
        assert eda.corr_train == None
        assert eda.corr_validation == None
        assert eda.corr_test == None

        assert eda.corr.shape == (3,3)

        eda.compute_corr(split='train', regex='^column_[1-3]$')

        assert eda.corr is not None
        assert eda.corr_train is not None
        assert eda.corr_validation == None
        assert eda.corr_test == None

        assert eda.corr_train.shape == (3,3)

        eda.compute_corr(split='validation', regex='^column_[1-3]$')

        assert eda.corr is not None
        assert eda.corr_train is not None
        assert eda.corr_validation is not None
        assert eda.corr_test == None

        assert eda.corr_validation.shape == (3,3)

        eda.compute_corr(split='test', regex='^column_[1-3]$')

        assert eda.corr is not None
        assert eda.corr_train is not None
        assert eda.corr_validation is not None
        assert eda.corr_test is not None

        assert eda.corr_test.shape == (3,3)

    def test_group_stats(self):

        print('\nTesting grouping statistics...')

        eda = EDA(data)
        group = eda.group_stats(by='column_1', columns=['column_2', 'column_3'])

        assert isinstance(group, pd.DataFrame)
