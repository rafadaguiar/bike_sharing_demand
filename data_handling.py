from datetime import datetime
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load(modify_dfs=True, load_=False):
    """
    Loads the train and test datasets. A dataframe with variable name
    folowed by an underscore (df_) has numerical features scaled and
    categorical features transformed to dummies.

    Parameters
    ----------
    modify_dfs True if modifications on the dataframes are intended
    load_           True if df_ is needed
    """
    if modify_dfs:
        print "Reading original files..."
        df_train = pd.read_csv('data/train.csv')
        df_test = pd.read_csv('data/test.csv')
        train, test, train_, test_, mms = preprocess(
            df_train,
            df_test
        )

        train.to_pickle("data/modified_train.pkl")
        test.to_pickle("data/modified_test.pkl")

        out_train_ = open("data/modified_train_.pkl", "wb")
        out_test_ = open("data/modified_test_.pkl", "wb")
        pickle.dump((train_, mms), out_train_)
        pickle.dump(test_, out_test_)
        out_train_.close(), out_test_.close()
    else:
        print "Reading pre-processed files..."
        train = pd.read_pickle("data/modified_train.pkl")
        test = pd.read_pickle("data/modified_test.pkl")
        if not load_:
            return train, test
        else:
            in_train_ = open("data/modified_train_.pkl", "rb")
            train_, mms = pickle.load(in_train_)
            in_test_ = open("data/modified_test_.pkl", "rb")
            test_ = pickle.load(in_test_)
            in_train_.close(), in_test_.close()
    return train, test, train_, test_, mms


def preprocess(train, test):
    """Performs the folowing dataset transformations: change data type,
    log transformation, addition of grouped variable, removal of unwanted
    features, scaling of numerical features and transformation to dummies
    of categorical features.
    """
    print "Preprocessing Dataset..."

    def modify_features(df):
        """Change variable types and apply log tranformations to dependent
        variables.
        """
        date_time = pd.DatetimeIndex(df['datetime'])

        df['day'] = date_time.day.astype('int64')
        df['weekday'] = date_time.dayofweek
        df['hour'] = date_time.hour
        df['month'] = date_time.month
        df['year'] = date_time.year

        df['humidity'] = df['humidity'].astype('float')
        df['weather'] = df['weather'].astype('int32')
        df['season'] = df['season'].astype('int32')
        df['holiday'] = df['holiday'].astype('int32')
        df['workingday'] = df['workingday'].astype('int32')

        if df.get('count') is not None:
            df['casual'] = np.log(df['casual']+1)
            df['registered'] = np.log(df['registered']+1)
        return df

    def add_grouped_count(df):
            """Creates grouped features based on aggregating a dependent
            variable over some key.
            """
            dependent_variables = ['count', 'casual', 'registered']
            groupby_key_list = [
                'year',
                'season',
                'weather',
                'month',
                'hour'
            ]
            for var in dependent_variables:
                for groupby_key in groupby_key_list:
                    grouped_count = train.groupby(groupby_key)[[var]]\
                        .agg(sum)
                    grouped_count.columns = [var+'.by.'+groupby_key]
                    grouped_count = grouped_count.astype('float')
                    df = df.join(grouped_count, on=groupby_key)
            return df

    def clean_n_scale(df):
        """Removes unwanted fields and scale variables for use on neural
        networks.
        """
        dependent = ['casual', 'registered']
        numerical = df.columns[df.dtypes == 'float'].tolist()

        if 'count' in df.columns:
            numerical = [
                var for var in numerical
                if var not in dependent and var != 'count'
            ]
        df[numerical] = df[numerical].apply(
            lambda x: MinMaxScaler().fit_transform(x)
        )

        index = df['datetime']
        df = df.drop(['datetime'], axis=1)
        if 'count' in df.columns:
            mms = dict()
            for var in dependent:
                mms[var] = MinMaxScaler()
                df[var] = mms[var].fit_transform(df[var])
            return df, mms
        else:
            return df

    def create_dummies(df):
        """Categorical variables are tranformed in dummies variables for use
        on neural networks.
        """
        categorical = df.columns[df.dtypes == 'int32'].tolist()
        for var in categorical:
            df = df.join(
                pd.get_dummies(df[var], prefix=var)
                .astype('int32')
            )
        df = df.drop(categorical, axis=1)
        return df

    train = add_grouped_count(modify_features(train))
    test = add_grouped_count(modify_features(test))
    train_, mms = clean_n_scale(create_dummies(train))
    test_ = clean_n_scale(create_dummies(test))
    return train, test, train_, test_, mms


def output(filename, prediction, test):
    """Creates the submission file for the Kaggle competition.
    """
    print """\
    --------------------------------------------------------------
    >>>>>>>>>>>>>>>> Generating new submission... <<<<<<<<<<<<<<<<
    >>>>>>>>>>>>>>>>           %s         <<<<<<<<<<<<<<<<
    --------------------------------------------------------------\
    """ % filename
    prediction = pd.DataFrame(
        data=prediction,
        dtype=int,
        index=test['datetime'],
        columns=['count']
    )
    pd.DataFrame.to_csv(prediction, 'data/%s' % filename)
