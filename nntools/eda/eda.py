from nntools import *


class EDA:
    '''
    Perform basic exploratory data analysis (EDA) tasks.
    '''

    def __init__(self, data):
        '''
        Arguments:

            data: the DataStructure containing the dataframe to analyse.
        '''

        self.data            = data # data structure
        self.corr            = None # correlations (full)
        self.corr_train      = None # correlations (train)
        self.corr_validation = None # correlations (validation)
        self.corr_test       = None # correlations (test)

    def compute_corr(self, split='train', regex=None):
        '''
        Compute the correlation of the features.

        Arguments:

            split: one of ['full', 'train', 'validation', 'test'],
            regex: regex to filter the desired columns.
        '''

        # select split
        if split == 'full':
            df = self.data.df

            # select columns if necessary
            if regex is not None:
                df = df.filter(regex=regex)

            # compute correlations
            self.corr = df.corr()

        if split == 'train':
            df = self.data.df_train

            # select columns if necessary
            if regex is not None:
                df = df.filter(regex=regex)

            # compute correlations
            self.corr_train = df.corr()

        if split == 'validation':
            df = self.data.df_validation

            # select columns if necessary
            if regex is not None:
                df = df.filter(regex=regex)

            # compute correlations
            self.corr_validation = df.corr()

        if split == 'test':
            df = self.data.df_test

            # select columns if necessary
            if regex is not None:
                df = df.filter(regex=regex)

            # compute correlations
            self.corr_test = df.corr()

        return self

    def group_stats(self, by, columns=None, agg=['count', 'mean', 'std', 'min', 'median', 'max'], split='full'):
        '''
        Group statistics by columns.

        Arguments:

            by:      group by these variables,
            columns: select only these columns (or the entire dataframe),
            agg:     list of functions to show,
            split:   one of ['full', 'train', 'validation', 'test'].
        '''
        
        if not isinstance(by, list):
            by = [by]
        if isinstance(columns, str):
            columns = [columns]

        # select the dataset
        if columns is not None:
            # sanitise the input
            for var in by:
                if var not in columns:
                    columns.append(var)

            # select
            if split == 'full':
                df = self.data.df[columns]
            if split == 'train':
                df = self.data.df_train[columns]
            if split == 'validation':
                df = self.data.df_validation[columns]
            if split == 'test':
                df = self.data.df_test[columns]
        else:
            if split == 'full':
                df = self.data.df
            if split == 'train':
                df = self.data.df_train
            if split == 'validation':
                df = self.data.df_validation
            if split == 'test':
                df = self.data.df_test

        # group and show statistics
        return df.groupby(by=by).agg(agg)

