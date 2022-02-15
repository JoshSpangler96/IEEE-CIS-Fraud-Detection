from sklearn.preprocessing import LabelEncoder
import datetime
import logging
import numpy as np
import pandas as pd

def cat_feature_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical Feature Encoding to prepare the dataset for training

    :param
    df: pd.DataFrame
        Dataframe for categorical feature encoding to be applied to
    :return:
    pd.DataFrame
    """
    # get a list of all categorical features in the datasets
    d_type = pd.DataFrame(df.dtypes)
    d_type.rename({0: 'dtype'}, axis=1, inplace=True)
    d_type.reset_index(inplace=True)
    d_type['dtype'] = d_type['dtype'].astype('string')
    cat_col = d_type.loc[(d_type['dtype'] == 'object')]['index'].to_list()

    # Add categorical features that are not objects
    # Listed in the data overview section of the Kaggle competition
    card_cat = [f'card{x}' for x in range(1, 7)]
    addr_cat = [f'addr{x}' for x in range(1, 3)]
    id_cat = [f'id_{x}' for x in range(12, 39)]
    cat_col += card_cat + addr_cat + id_cat
    cat_col = list(set(cat_col))

    for i in range(len(cat_col)):
        if cat_col[i] in df.columns:
            le = LabelEncoder()
            le.fit(list(df[cat_col[i]].astype(str).values) + list(df[cat_col[i]].astype(str).values))
            df[cat_col[i]] = le.transform(list(df[cat_col[i]].astype(str).values))

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for IEEE Fraud Detection Dataset

    :param
    df: pd.DataFrame
        Dataframe for feature engineering to be applied to

    :return:
    pd.DataFrame
    """
    startdate = datetime.datetime.strptime('2017-12-01', "%Y-%m-%d")

    df['cents'] = (df['TransactionAmt'] - np.floor(df['TransactionAmt'])).astype('float32')
    df['log_TransactionAmt'] = np.log(df['TransactionAmt'])
    df[['P_email_1', 'P_email_2', 'P_email_3']] = df['P_emaildomain'].str.split('.', expand=True)
    df.drop('P_emaildomain', axis=1, inplace=True)
    df[['R_email_1', 'R_email_2', 'R_email_3']] = df['R_emaildomain'].str.split('.', expand=True)
    df.drop('R_emaildomain', axis=1, inplace=True)
    df['Date'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
    df['Weekdays'] = df['Date'].dt.dayofweek
    df['Hours'] = df['Date'].dt.hour
    df['Days'] = df['Date'].dt.day
    df['Month'] = (df['Date'].dt.year - 2017) * 12 + df['Date'].dt.month

    # Aggregate features together
    df['uid'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
    df['uid2'] = df['uid'].astype(str) + '_' + np.floor(df.Days - df.D1).astype(str)
    for feature in ['TransactionAmt', 'D9', 'D10', 'C1', 'C5']:
        for agg in ['mean', 'std']:
            temp = df.groupby(['uid2'])[feature].agg([agg]).reset_index().rename(
                columns={f'{agg}': f'{feature}_{agg}_uid'})
            temp.index = list(temp['uid2'])
            temp = temp[f'{feature}_{agg}_uid'].to_dict()
            df[f'{feature}_{agg}_uid'] = df['uid2'].map(temp).astype('float32')
    df.drop('uid2', axis=1, inplace=True)

    for feature in ['card1', 'card4']:
        for agg in ['mean', 'std']:
            df[f'TransactionAmt_{agg}_{feature}'] = df['TransactionAmt'] / df.groupby([feature])[
                'TransactionAmt'].transform(agg)
            df[f'id_01_{agg}_{feature}'] = df['TransactionAmt'] / df.groupby([feature])['id_01'].transform(agg)

    return df

def reduce_mem_usage2(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    logging.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    logging.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logging.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df