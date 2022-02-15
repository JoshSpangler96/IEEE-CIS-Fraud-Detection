import lightgbm as lgb
import pandas as pd
from imblearn.over_sampling import SMOTE
import joblib


def run_lightgbm(X: pd.DataFrame, y: pd.DataFrame, split_percent: float):
    """
    Create a LightGBM model for the IEEE Fraud Dataset

    :param X: Matrix of features to be trained
    :param y: Target variable
    :param split_percent: percentage of data to be used as training (e.g. 0.80 = 80%)
    :return:
    """

    # split into train and test data
    trn_idx = int(split_percent * len(X))
    X_train = X[:trn_idx]
    y_train = y[:trn_idx]
    X_test = X[trn_idx:]
    y_test = y[trn_idx:]

    smote = SMOTE()
    X_sm, y_sm = smote.fit_resample(X_train, y_train)

    hyper_parameters = {
        'num_leaves': 512,
        'min_child_weight': 0.001,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.75,
        'min_data_in_leaf': 106,
        'objective': 'binary',
        'learning_rate': 0.02,
        'boosting_type': 'gbdt',
        'metric': 'auc',
        "verbosity": -1,
        'reg_lambda': 0.4,
        'reg_alpha': 0.6,
    }

    trn_data = lgb.Dataset(X_sm, label=y_sm)
    val_data = lgb.Dataset(X_test, label=y_test)

    lgb_clf = lgb.train(
        hyper_parameters,
        trn_data,
        10000,
        valid_sets=[trn_data, val_data],
        verbose_eval=200,
        early_stopping_rounds=200
    )

    joblib.dump(lgb_clf, '../model/lgb_model.pkl')

    return lgb_clf
