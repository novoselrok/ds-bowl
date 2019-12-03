import math
import os

import numpy as np
from bayes_opt import BayesianOptimization, JSONLogger, Events
from bayes_opt.util import load_logs
from sklearn.metrics import roc_auc_score, mean_squared_error

from submission import get_train_test_features, label_encode_categorical_features, stratified_group_k_fold, fit_model, \
    integer_encode_params, get_lgbm_classifier, get_lgbm_regressor, binarize_accuracy_group, get_catboost_classifier, \
    categorical_features, get_catboost_regressor

np.random.seed(0)


def cv(model_fn, score_fn, fit_fn, X, y, y_accuracy_group, installation_ids, model_params,
       n_splits=5, predict_proba=False):
    scores = []

    kf = stratified_group_k_fold(None, y_accuracy_group, installation_ids, n_splits, seed=2019)
    for fold, (train_split, test_split) in enumerate(kf):
        X_train, X_test = X.iloc[train_split, :], X.iloc[test_split, :]
        y_train, y_test = y[train_split], y[test_split]

        model = model_fn(model_params)
        fit_fn(model, X_train, y_train, X_test, y_test)
        if predict_proba:
            pred = model.predict_proba(X_test)
        else:
            pred = model.predict(X_test)

        score = score_fn(y_test, pred)
        scores.append(score)

    return np.mean(scores)


def lgb_cv_accuracy_group_ordinal_model(target):
    def _eval(**params):
        features = get_train_test_features()

        df_train, _ = label_encode_categorical_features(features['df_train_features'])
        y_accuracy_group = binarize_accuracy_group(features['y_accuracy_group'], target)

        model_params = integer_encode_params(params)
        score = cv(
            get_lgbm_classifier('binary', 'auc'),
            roc_auc_score,
            fit_model,
            df_train,
            y_accuracy_group,
            y_accuracy_group,
            features['train_installation_ids'],
            model_params
        )

        print(f'CV score: {score}')
        return score

    return _eval


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


def catboost_cv_accuracy_group_ordinal_model(target):
    def _eval(**params):
        features = get_train_test_features()

        df_train, _ = label_encode_categorical_features(features['df_train_features'])
        cat_features = [list(df_train.columns).index(feature) for feature in categorical_features]
        y_accuracy_group = binarize_accuracy_group(features['y_accuracy_group'], target)

        model_params = integer_encode_params(params)
        score = cv(
            get_catboost_classifier('Logloss', 'AUC', cat_features),
            roc_auc_score,
            fit_model,
            df_train,
            y_accuracy_group,
            y_accuracy_group,
            features['train_installation_ids'],
            model_params
        )

        print(f'CV score: {score}')
        return score

    return _eval


def bayes_opt(fn, name, params, probes=None, n_iter=80, init_points=20):
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
    CV_ORDINAL_CLASSIFICATION = True
    CV_ACCURACY_RATE_REGRESSION = True
    CV_ACCURACY_GROUP_REGRESSION = True

    lgb_bayes_params = {
        'learning_rate': (0.01, 1.0),
        'max_depth': (3, 10),
        'max_bin': (2, 255),
        'num_leaves': (5, 255),
        'min_child_samples': (50, 1500),
        'min_child_weight': (0.1, 1000),
        'colsample_bytree': (0.1, 0.66),
        'subsample': (0.7, 0.9),
        'reg_alpha': (1, 30),
        'reg_lambda': (1, 30),
    }

    if CV_CORRECT_ATTEMPTS:
        bayes_opt(lgb_cv_correct_attempts, 'lgb_cv_correct_attempts', lgb_bayes_params)

    if CV_ORDINAL_CLASSIFICATION:
        for target in [0, 1, 2]:
            bayes_opt(
                lgb_cv_accuracy_group_ordinal_model(target), f'lgb_cv_ordinal_{target}', lgb_bayes_params)

    if CV_ACCURACY_RATE_REGRESSION:
        bayes_opt(lgb_cv_accuracy_rate_regression, 'lgb_cv_accuracy_rate_regression', lgb_bayes_params)

    if CV_ACCURACY_GROUP_REGRESSION:
        bayes_opt(lgb_cv_accuracy_group_regression, 'lgb_cv_accuracy_group_regression', lgb_bayes_params)


def catboost_tuning():
    CV_CORRECT_ATTEMPTS = True
    CV_ORDINAL_CLASSIFICATION = True
    CV_ACCURACY_RATE_REGRESSION = True
    CV_ACCURACY_GROUP_REGRESSION = True

    catboost_bayes_params = {
        'learning_rate': (0.01, 1.0),
        'max_depth': (3, 8),
        'l2_leaf_reg': (1, 30),
        'colsample_bylevel': (0.1, 0.7),
    }

    if CV_CORRECT_ATTEMPTS:
        bayes_opt(catboost_cv_correct_attempts, 'catboost_cv_correct_attempts', catboost_bayes_params)

    if CV_ORDINAL_CLASSIFICATION:
        for target in [0, 1, 2]:
            bayes_opt(
                catboost_cv_accuracy_group_ordinal_model(target), f'catboost_ordinal_{target}', catboost_bayes_params)

    if CV_ACCURACY_RATE_REGRESSION:
        bayes_opt(catboost_cv_accuracy_rate_regression, 'catboost_cv_accuracy_rate_regression', catboost_bayes_params)

    if CV_ACCURACY_GROUP_REGRESSION:
        bayes_opt(catboost_cv_accuracy_group_regression, 'catboost_cv_accuracy_group_regression', catboost_bayes_params)


if __name__ == '__main__':
    lgb_tuning()
    catboost_tuning()
