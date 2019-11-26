import math
import os

import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization, JSONLogger, Events
from bayes_opt.util import load_logs
from sklearn.metrics import roc_auc_score, mean_squared_error, cohen_kappa_score

from submission import get_train_test_features, label_encode_categorical_features, stratified_group_k_fold, fit_model, \
    integer_encode_params, get_lgbm_classifier, get_lgbm_regressor


def cv(model_fn, score_fn, X, y, y_accuracy_group, installation_ids, model_params, n_splits=5, predict_proba=False):
    scores = []

    kf = stratified_group_k_fold(None, y_accuracy_group, installation_ids, n_splits, seed=2019)
    for fold, (train_split, test_split) in enumerate(kf):
        X_train, X_test = X.iloc[train_split, :], X.iloc[test_split, :]
        y_train, y_test = y[train_split], y[test_split]

        model = model_fn(model_params)
        fit_model(model, X_train, y_train, X_test, y_test)
        if predict_proba:
            pred = model.predict_proba(X_test)
        else:
            pred = model.predict(X_test)

        score = score_fn(y_test, pred)
        scores.append(score)

    return np.mean(scores)


def cv_accuracy_group_ordinal_model(target):
    def _eval(**params):
        (
            df_train_features,
            _,
            _,
            _,
            _,
            y_accuracy_group,
            train_installation_ids,
            _
        ) = get_train_test_features()

        pos_label_indices = y_accuracy_group > target
        neg_label_indices = y_accuracy_group <= target
        y_accuracy_group[pos_label_indices] = 1
        y_accuracy_group[neg_label_indices] = 0

        df_train, _ = label_encode_categorical_features(df_train_features)

        model_params = integer_encode_params(params)
        score = cv(
            get_lgbm_classifier('binary', 'auc'),
            roc_auc_score,
            df_train,
            y_accuracy_group,
            y_accuracy_group,
            train_installation_ids,
            model_params
        )

        print(f'CV score: {score}')
        return score

    return _eval


def cv_accuracy_group_classification(**params):
    (
        df_train_features,
        _,
        _,
        _,
        _,
        y_accuracy_group,
        train_installation_ids,
        _
    ) = get_train_test_features()

    df_train, _ = label_encode_categorical_features(df_train_features)

    model_params = integer_encode_params(params)
    score = cv(
        get_lgbm_classifier('multiclass', 'multi_logloss'),
        lambda y_true, y_pred: cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        df_train,
        y_accuracy_group,
        y_accuracy_group,
        train_installation_ids,
        model_params
    )

    print(f'CV score: {score}')
    return score


def cv_correct_attempts(**params):
    (
        df_train_features,
        _,
        y_correct,
        _,
        _,
        y_accuracy_group,
        train_installation_ids,
        _
    ) = get_train_test_features()

    df_train, _ = label_encode_categorical_features(df_train_features)
    model_params = integer_encode_params(params)

    score = cv(
        get_lgbm_classifier('binary', 'auc'),
        roc_auc_score,
        df_train,
        y_correct,
        y_accuracy_group,
        train_installation_ids,
        model_params
    )

    print(f'CV score: {score}')
    return score


def cv_accuracy_group_regression(**params):
    (
        df_train_features,
        _,
        _,
        _,
        _,
        y_accuracy_group,
        train_installation_ids,
        _
    ) = get_train_test_features()

    df_train, _ = label_encode_categorical_features(df_train_features)
    model_params = integer_encode_params(params)

    score = cv(
        get_lgbm_regressor('regression', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        df_train,
        y_accuracy_group,
        y_accuracy_group,
        train_installation_ids,
        model_params
    )

    print(f'CV score: {score}')
    return -score


def cv_accuracy_rate_regression(**params):
    (
        df_train_features,
        _,
        _,
        _,
        y_accuracy_rate,
        y_accuracy_group,
        train_installation_ids,
        _
    ) = get_train_test_features()

    df_train, _ = label_encode_categorical_features(df_train_features)
    model_params = integer_encode_params(params)

    score = cv(
        get_lgbm_regressor('regression', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        df_train,
        y_accuracy_rate,
        y_accuracy_group,
        train_installation_ids,
        model_params
    )

    print(f'CV score: {score}')
    return -score


def bayes_opt(fn, name, params, probes=None):
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

    opt.maximize(n_iter=60, init_points=20)
    print(opt.max)


def main():
    CV_ACCURACY_GROUP_CLASSIFICATION = False
    CV_ORDINAL_CLASSIFICATION = False
    CV_CORRECT_ATTEMPTS = True
    CV_ACCURACY_GROUP_REGRESSION = True
    CV_ACCURACY_RATE_REGRESSION = False

    lgb_bayes_params = {
        'learning_rate': (0.01, 1.0),
        'max_depth': (3, 16),
        'max_bin': (2, 500),
        'num_leaves': (5, 500),
        'min_child_samples': (50, 1500),
        'min_child_weight': (0.1, 1000),
        'colsample_bytree': (0.05, 0.6),
        'subsample': (0.7, 0.99),
        'reg_alpha': (1, 30),
        'reg_lambda': (1, 30),
    }

    if CV_ACCURACY_GROUP_CLASSIFICATION:
        bayes_opt(cv_accuracy_group_classification, 'cv_accuracy_group_classification', lgb_bayes_params)

    if CV_ORDINAL_CLASSIFICATION:
        for target in [0, 1, 2]:
            bayes_opt(
                cv_accuracy_group_ordinal_model(target), f'ordinal_{target}', lgb_bayes_params)

    if CV_ACCURACY_RATE_REGRESSION:
        bayes_opt(cv_accuracy_rate_regression, 'cv_accuracy_rate_regression', lgb_bayes_params)

    if CV_CORRECT_ATTEMPTS:
        bayes_opt(cv_correct_attempts, 'cv_correct_attempts', lgb_bayes_params)

    if CV_ACCURACY_GROUP_REGRESSION:
        bayes_opt(cv_accuracy_group_regression, 'cv_accuracy_group_regression', lgb_bayes_params)


if __name__ == '__main__':
    main()
