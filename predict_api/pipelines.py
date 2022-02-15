import logging
import pandas as pd
import numpy as np
from resources import cat_feature_encoding, feature_engineering, reduce_mem_usage2
import model


def ieee_train_pipeline(identity_path: str, transaction_path: str):
    """
    Data pipeline for IEEE Fraud Training Dataset

    :parameter
    identity_path: str
        path to file with identity data
    transaction_path: str
        path to file with transaction data
    """

    logging.info('Starting the data analysis pipeline')

    # load the csv files into memmory
    identity = pd.read_csv(identity_path)
    transaction = pd.read_csv(transaction_path)

    # merge the testing and training data
    df = pd.merge(identity, transaction, on='TransactionID', how='left')
    df.columns = df.columns.str.replace('[-]', '_', regex=True)

    del identity, transaction

    # features_to_use determined from EDA (Path to EDA: ../explore/IEEE_Fraud_Detection_EDA.ipynb)
    features_to_use = [
        'TransactionID', 'TransactionDT', 'TransactionAmt',
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'DeviceType',
        'DeviceInfo', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
        'V147', 'V142', 'V1', 'V174', 'V109', 'V201', 'V238', 'V271', 'V78',
        'V160', 'V65', 'V339', 'V27', 'V138', 'V320', 'V6', 'V223', 'V114',
        'V118', 'V173', 'V80', 'V107', 'V258', 'V44', 'V198', 'V252', 'V220',
        'V309', 'V209', 'V67', 'V124', 'V260', 'V155', 'V176', 'V55', 'V36',
        'V325', 'V127', 'V175', 'V82', 'V20', 'V329', 'V111', 'V139', 'V210',
        'V30', 'V86', 'V3', 'V37', 'V13', 'V207', 'V286', 'V47', 'V162', 'V8',
        'V62', 'V234', 'V56', 'V240', 'V23', 'V4', 'V115', 'V166', 'V121', 'V76',
        'V259', 'V312', 'V120', 'V169', 'V305', 'V291', 'V185', 'V26', 'V241',
        'V250', 'V108', 'V261', 'V54', 'D5', 'D14', 'D1', 'D9', 'D13', 'D8',
        'D15', 'D10', 'C1', 'C5', 'C3', 'M5', 'M7', 'M4', 'M2', 'M9', 'M6',
        'M1', 'M3', 'M8', 'id_08', 'id_09', 'id_28', 'id_07', 'id_37', 'id_01',
        'id_27', 'id_25', 'id_21', 'id_12', 'id_38', 'id_35', 'id_30', 'id_04',
        'id_05', 'id_29', 'id_36', 'id_34', 'id_11', 'id_26', 'id_13', 'id_24',
        'id_23', 'id_20', 'id_17', 'id_14', 'id_31', 'id_32', 'id_03', 'id_10',
        'id_18', 'id_02', 'id_06', 'id_16', 'id_15', 'id_19', 'id_33', 'id_22'
    ]

    df = df[features_to_use + ['isFraud']]
    # add feature engineering
    df = feature_engineering(df)
    # encode categorical features
    df = cat_feature_encoding(df)
    # handling missing data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(-1, inplace=True)

    # seperate the data into a matrix of features (X) and target variable (y)
    X = df.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'Date'], axis=1)
    y = df.sort_values('TransactionDT')['isFraud']

    # reduce the memory usage
    X = reduce_mem_usage2(X)
    del df

    # create and train model
    lgb_model = model.run_lightgbm(X, y, 0.8)
    return lgb_model


def ieee_test_pipeline(identity_path: str, transaction_path: str) -> pd.DataFrame:
    """
    Data pipeline for IEEE Fraud Test Dataset

    :parameter
    identity_path: str
        path to file with identity data
    transaction_path: str
        path to file with transaction data

    :return: pd.DataFrame
    """

    logging.info('Starting the data analysis pipeline')

    # load the csv files into memmory
    identity = pd.read_csv(identity_path)
    transaction = pd.read_csv(transaction_path)

    # merge the testing and training data
    df = pd.merge(identity, transaction, on='TransactionID', how='left')
    df.columns = df.columns.str.replace('[-]', '_', regex=True)

    del identity, transaction

    # features_to_use determined from EDA (Path to EDA: ../explore/IEEE_Fraud_Detection_EDA.ipynb)
    features_to_use = [
        'TransactionID', 'TransactionDT', 'TransactionAmt',
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'DeviceType',
        'DeviceInfo', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
        'V147', 'V142', 'V1', 'V174', 'V109', 'V201', 'V238', 'V271', 'V78',
        'V160', 'V65', 'V339', 'V27', 'V138', 'V320', 'V6', 'V223', 'V114',
        'V118', 'V173', 'V80', 'V107', 'V258', 'V44', 'V198', 'V252', 'V220',
        'V309', 'V209', 'V67', 'V124', 'V260', 'V155', 'V176', 'V55', 'V36',
        'V325', 'V127', 'V175', 'V82', 'V20', 'V329', 'V111', 'V139', 'V210',
        'V30', 'V86', 'V3', 'V37', 'V13', 'V207', 'V286', 'V47', 'V162', 'V8',
        'V62', 'V234', 'V56', 'V240', 'V23', 'V4', 'V115', 'V166', 'V121', 'V76',
        'V259', 'V312', 'V120', 'V169', 'V305', 'V291', 'V185', 'V26', 'V241',
        'V250', 'V108', 'V261', 'V54', 'D5', 'D14', 'D1', 'D9', 'D13', 'D8',
        'D15', 'D10', 'C1', 'C5', 'C3', 'M5', 'M7', 'M4', 'M2', 'M9', 'M6',
        'M1', 'M3', 'M8', 'id_08', 'id_09', 'id_28', 'id_07', 'id_37', 'id_01',
        'id_27', 'id_25', 'id_21', 'id_12', 'id_38', 'id_35', 'id_30', 'id_04',
        'id_05', 'id_29', 'id_36', 'id_34', 'id_11', 'id_26', 'id_13', 'id_24',
        'id_23', 'id_20', 'id_17', 'id_14', 'id_31', 'id_32', 'id_03', 'id_10',
        'id_18', 'id_02', 'id_06', 'id_16', 'id_15', 'id_19', 'id_33', 'id_22'
    ]

    df = df[features_to_use]
    # add feature engineering
    df = feature_engineering(df)
    # encode categorical features
    df = cat_feature_encoding(df)
    # handling missing data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(-1, inplace=True)
    # remove date fields
    df.drop(['TransactionDT', 'Date'], axis=1, inplace=True)

    return df
