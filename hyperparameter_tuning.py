import math
import os

import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization, JSONLogger, Events
from bayes_opt.util import load_logs
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, mean_squared_error, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

from submission import get_train_test_features, label_encode_categorical_features, stratified_group_k_fold, fit_model, \
    integer_encode_params, get_lgbm_classifier, get_lgbm_regressor, get_catboost_classifier, \
    categorical_features, get_catboost_regressor, \
    ohe_encode_categorical_features, get_xgboost_regressor, get_xgboost_classifier

np.random.seed(0)


def cv(model_fn, score_fn, fit_fn, X, y, y_accuracy_group, installation_ids, model_params,
       n_splits=10, predict_proba=False, standard_scale=False, pca_transform=False):
    scores = []

    (train_split, val_split) = list(
        stratified_group_k_fold(None, y_accuracy_group, installation_ids, n_splits, seed=0))[0]

    X, X_val = X.iloc[train_split, :], X.iloc[val_split, :]
    y, y_val = y[train_split], y[val_split]
    y_accuracy_group, installation_ids = y_accuracy_group[train_split], installation_ids[train_split]

    oof_val_pred = np.zeros(X_val.shape[0])

    kf = stratified_group_k_fold(None, y_accuracy_group, installation_ids, n_splits, seed=2019)
    for fold, (train_split, test_split) in enumerate(kf):
        X_train, X_test = X.iloc[train_split, :], X.iloc[test_split, :]
        y_train, y_test = y[train_split], y[test_split]

        model = model_fn(model_params)
        fit_fn(model, X_train, y_train, X_test, y_test)
        if predict_proba:
            pred = model.predict_proba(X_test)[:, 1]
            val_pred = model.predict_proba(X_val)[:, 1]
        else:
            pred = model.predict(X_test)
            val_pred = model.predict(X_val)

        oof_val_pred += (val_pred / n_splits)
        score = score_fn(y_test, pred)
        scores.append(score)

    val_score = score_fn(y_val, oof_val_pred)

    print(f'Mean CV score: {np.mean(scores)}, val score: {val_score}')

    return val_score


def lgb_cv_correct_attempts(**params):
    features = get_train_test_features()

    df_train, _ = label_encode_categorical_features(features['df_train_features'])
    model_params = integer_encode_params(params)

    score = cv(
        get_lgbm_classifier('binary', 'auc'),
        roc_auc_score,
        fit_model,
        df_train,
        features['y_correct'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return score


def lgb_cv_accuracy_group_regression(**params):
    features = get_train_test_features()

    df_train, _ = label_encode_categorical_features(features['df_train_features'])
    model_params = integer_encode_params(params)

    score = cv(
        get_lgbm_regressor('regression', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        features['y_accuracy_group'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return -score


def lgb_cv_accuracy_rate_regression(**params):
    features = get_train_test_features()

    df_train, _ = label_encode_categorical_features(features['df_train_features'])
    model_params = integer_encode_params(params)

    score = cv(
        get_lgbm_regressor('regression', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        features['y_accuracy_rate'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return -score


def catboost_cv_correct_attempts(**params):
    features = get_train_test_features()

    df_train, _ = label_encode_categorical_features(features['df_train_features'])
    cat_features = [list(df_train.columns).index(feature) for feature in categorical_features]
    model_params = integer_encode_params(params)

    score = cv(
        get_catboost_classifier('Logloss', 'AUC', cat_features),
        roc_auc_score,
        fit_model,
        df_train,
        features['y_correct'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return score


def catboost_cv_accuracy_group_regression(**params):
    features = get_train_test_features()

    df_train, _ = label_encode_categorical_features(features['df_train_features'])
    cat_features = [list(df_train.columns).index(feature) for feature in categorical_features]
    model_params = integer_encode_params(params)

    score = cv(
        get_catboost_regressor('RMSE', 'RMSE', cat_features),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        features['y_accuracy_group'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return -score


def catboost_cv_accuracy_rate_regression(**params):
    features = get_train_test_features()

    df_train, _ = label_encode_categorical_features(features['df_train_features'])
    cat_features = [list(df_train.columns).index(feature) for feature in categorical_features]
    model_params = integer_encode_params(params)

    score = cv(
        get_catboost_regressor('RMSE', 'RMSE', cat_features),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        features['y_accuracy_rate'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return -score


def xgboost_cv_correct_attempts(**params):
    features = get_train_test_features()

    df_train, _ = ohe_encode_categorical_features(features['df_train_features'])
    model_params = integer_encode_params(params)

    score = cv(
        get_xgboost_classifier('binary:logistic', 'auc'),
        roc_auc_score,
        fit_model,
        df_train,
        features['y_correct'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return score


def xgboost_cv_accuracy_group_regression(**params):
    features = get_train_test_features()

    df_train, _ = ohe_encode_categorical_features(features['df_train_features'])
    model_params = integer_encode_params(params)

    score = cv(
        get_xgboost_regressor('reg:squarederror', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        features['y_accuracy_group'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return -score


def xgboost_cv_accuracy_rate_regression(**params):
    features = get_train_test_features()

    df_train, _ = ohe_encode_categorical_features(features['df_train_features'])
    model_params = integer_encode_params(params)

    score = cv(
        get_xgboost_regressor('reg:squarederror', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        fit_model,
        df_train,
        features['y_accuracy_rate'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    print(f'CV score: {score}')
    return -score


def bayes_opt(fn, name, params, probes=None, n_iter=40, init_points=10):
    print(f'Optimizing {name}...')
    opt = BayesianOptimization(fn, params, verbose=2)
    if os.path.exists(f'./bayes_opt_logs/{name}.json'):
        print('Loading logs...')
        load_logs(opt, logs=[f'./bayes_opt_logs/{name}.json'])

    logger = JSONLogger(path=f'./bayes_opt_logs/{name}.json')
    opt.subscribe(Events.OPTMIZATION_STEP, logger)

    # Probe with a set of know "good" params
    if probes:
        for probe in probes:
            opt.probe(params=probe, lazy=True)

    opt.maximize(n_iter=n_iter, init_points=init_points)
    print(opt.max)


def lgb_tuning():
    CV_CORRECT_ATTEMPTS = False
    CV_ACCURACY_RATE_REGRESSION = True
    CV_ACCURACY_GROUP_REGRESSION = True

    lgb_bayes_params = {
        'learning_rate': (0.01, 1.0),
        'max_depth': (3, 15),
        'max_bin': (2, 750),
        'num_leaves': (5, 750),
        'min_child_samples': (50, 1500),
        'min_child_weight': (0.1, 1000),
        'reg_alpha': (1, 200),
        'reg_lambda': (1, 200),
    }

    if CV_ACCURACY_GROUP_REGRESSION:
        bayes_opt(lgb_cv_accuracy_group_regression, 'lgb_cv_accuracy_group_regression', lgb_bayes_params,
                  n_iter=80, init_points=20)

    if CV_CORRECT_ATTEMPTS:
        bayes_opt(lgb_cv_correct_attempts, 'lgb_cv_correct_attempts', lgb_bayes_params,
                  n_iter=80, init_points=20)

    if CV_ACCURACY_RATE_REGRESSION:
        bayes_opt(lgb_cv_accuracy_rate_regression, 'lgb_cv_accuracy_rate_regression', lgb_bayes_params,
                  n_iter=80, init_points=20)


def catboost_tuning():
    CV_CORRECT_ATTEMPTS = False
    CV_ACCURACY_RATE_REGRESSION = True
    CV_ACCURACY_GROUP_REGRESSION = True

    catboost_bayes_params = {
        'learning_rate': (0.01, 1.0),
        'max_depth': (3, 10),
        'l2_leaf_reg': (1, 500),
    }

    if CV_CORRECT_ATTEMPTS:
        bayes_opt(catboost_cv_correct_attempts, 'catboost_cv_correct_attempts', catboost_bayes_params,
                  n_iter=80, init_points=20)

    if CV_ACCURACY_RATE_REGRESSION:
        bayes_opt(catboost_cv_accuracy_rate_regression, 'catboost_cv_accuracy_rate_regression', catboost_bayes_params,
                  n_iter=80, init_points=20)

    if CV_ACCURACY_GROUP_REGRESSION:
        bayes_opt(catboost_cv_accuracy_group_regression, 'catboost_cv_accuracy_group_regression', catboost_bayes_params,
                  n_iter=80, init_points=20)


def xgboost_tuning():
    CV_CORRECT_ATTEMPTS = False
    CV_ACCURACY_RATE_REGRESSION = True
    CV_ACCURACY_GROUP_REGRESSION = True

    xgboost_bayes_params = {
        'learning_rate': (0.01, 1.0),
        'max_depth': (3, 10),
        'max_bin': (2, 750),
        'min_child_samples': (50, 1500),
        'min_child_weight': (0.1, 1000),
        'reg_alpha': (1, 200),
        'reg_lambda': (1, 200),
    }

    if CV_ACCURACY_GROUP_REGRESSION:
        bayes_opt(xgboost_cv_accuracy_group_regression, 'xgboost_cv_accuracy_group_regression', xgboost_bayes_params,
                  n_iter=80, init_points=20)

    if CV_CORRECT_ATTEMPTS:
        bayes_opt(xgboost_cv_correct_attempts, 'xgboost_cv_correct_attempts', xgboost_bayes_params,
                  n_iter=80, init_points=20)

    if CV_ACCURACY_RATE_REGRESSION:
        bayes_opt(xgboost_cv_accuracy_rate_regression, 'xgboost_cv_accuracy_rate_regression', xgboost_bayes_params,
                  n_iter=80, init_points=20)


def meta_threshold_eval(feature, target):
    def inner(**params):
        bounds = [
            params['0_1'],
            params['1_2'],
            params['2_3']
        ]

        pred = pd.cut(feature, [-np.inf] + bounds + [np.inf], labels=[0, 1, 2, 3])

        score = cohen_kappa_score(pred, target, weights='quadratic')
        print(score)
        return score

    return inner


def meta_tuning():
    df_meta_train_features = pd.read_csv('preprocessed-data/meta_train_features.csv')
    target = df_meta_train_features['target'].values
    installation_ids = df_meta_train_features['installation_ids']

    df_train = df_meta_train_features.drop(columns=['target', 'installation_ids'])

    thresholds = {
        'correct_attempts': {
            '0_1': (0.0, 0.65),
            '1_2': (0.651, 0.79999),
            '2_3': (0.8, 1.0)
        },
        'group': {
            '0_1': (0.0, 1.4),
            '1_2': (1.401, 1.95),
            '2_3': (1.951, 3)
        },
        'rate': {
            '0_1': (0.0, 0.46),
            '1_2': (0.461, 0.65),
            '2_3': (0.651, 1.0)
        },
    }

    for feature in df_train.columns:
        if 'correct_attempts' in feature:
            params = thresholds['correct_attempts']
        elif 'group' in feature:
            params = thresholds['group']
        else:
            params = thresholds['rate']

        bayes_opt(
            meta_threshold_eval(df_meta_train_features[feature], target),
            f'meta_feature_thresholds_{feature}.json',
            params,
            init_points=100,
            n_iter=30
        )


if __name__ == '__main__':
    lgb_tuning()
    catboost_tuning()
    xgboost_tuning()
    # meta_tuning()
