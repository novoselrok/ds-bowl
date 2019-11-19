import math
import os
import time

import numpy as np
from bayes_opt import BayesianOptimization, JSONLogger, Events
from bayes_opt.util import load_logs
from sklearn.metrics import cohen_kappa_score, roc_auc_score, mean_squared_error
import matplotlib.pyplot as plt

from submission import get_train_test_features, stratified_group_k_fold, \
    label_encode_categorical_features, get_correct_attempts_model, fit_model, get_accuracy_group_model, \
    get_uncorrect_attempts_model

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

LGB_INTEGER_PARAMS = ['max_depth', 'max_bin', 'num_leaves', 'min_child_samples', 'n_splits', 'subsample_freq']


def integer_encode_params(params):
    for param, value in params.items():
        if param in LGB_INTEGER_PARAMS:
            params[param] = int(round(value))
    return params


def lgb_score(
        model, score_fn,
        X_train, _X_test,
        y_train, _y_test,
        y_test_accuracy_group, test_installation_ids
):
    scores = []
    kf = stratified_group_k_fold(None, y_test_accuracy_group, test_installation_ids, 2, seed=2019)
    for val_split, test_split in kf:
        X_val = _X_test.iloc[val_split]
        y_val = _y_test[val_split]

        X_test = _X_test.iloc[test_split]
        y_test = _y_test[test_split]

        fit_model(model, X_train, y_train, X_val, y_val)
        pred = model.predict(X_test)

        scores.append(score_fn(y_test, pred))

    return np.mean(scores)


def cv(model_fn, score_fn, X, y, y_accuracy_group, installation_ids, model_params, n_splits=5):
    scores = []

    kf = stratified_group_k_fold(None, y_accuracy_group, installation_ids, n_splits, seed=2019)
    for fold, (train_split, test_split) in enumerate(kf):
        X_train, X_test = X.iloc[train_split, :], X.iloc[test_split, :]
        y_train, y_test = y[train_split], y[test_split]

        score = lgb_score(
            model_fn(model_params), score_fn,
            X_train, X_test,
            y_train, y_test,
            y_accuracy_group[test_split], installation_ids[test_split]
        )
        print(f'Done fold: {fold}, Score: {score}')
        scores.append(score)

    time.sleep(5.0)
    return np.mean(scores)


def lgb_correct_attempts_eval(**params):
    (
        df_train_features,
        df_test_features,
        y_correct,
        _,
        y_accuracy_group,
        train_installation_ids,
        _
    ) = get_train_test_features()

    df_train, _ = label_encode_categorical_features(df_train_features, df_test_features)
    model_params = integer_encode_params(params)

    score = cv(
        get_correct_attempts_model,
        roc_auc_score,
        df_train,
        y_correct,
        y_accuracy_group,
        train_installation_ids,
        model_params
    )

    print(f'CV score: {score}')
    return score


def lgb_uncorrect_attempts_eval(**params):
    (
        df_train_features,
        df_test_features,
        _,
        y_uncorrect,
        y_accuracy_group,
        train_installation_ids,
        _
    ) = get_train_test_features()

    df_train, _ = label_encode_categorical_features(df_train_features, df_test_features)
    model_params = integer_encode_params(params)

    score = cv(
        get_uncorrect_attempts_model,
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        df_train,
        y_uncorrect,
        y_accuracy_group,
        train_installation_ids,
        model_params
    )

    print(f'CV score: {score}')
    return -score


def lgb_accuracy_group_eval(**params):
    (
        df_train_features,
        df_test_features,
        _,
        _,
        y_accuracy_group,
        train_installation_ids,
        _
    ) = get_train_test_features()

    df_train, _ = label_encode_categorical_features(df_train_features, df_test_features)
    model_params = integer_encode_params(params)

    score = cv(
        get_accuracy_group_model,
        lambda y_true, y_pred: cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        df_train,
        y_accuracy_group,
        y_accuracy_group,
        train_installation_ids,
        model_params
    )

    print(f'CV score: {score}')
    return score


def bayes_opt(fn, params, probes=None):
    name = fn.__name__
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

    opt.maximize(n_iter=100, init_points=60)
    print(opt.max)


if __name__ == '__main__':
    os.makedirs('bayes_opt_logs', exist_ok=True)

    bayes_opt(
        lgb_correct_attempts_eval,
        {
            'learning_rate': (0.01, 1.0),
            'max_depth': (3, 10),
            'max_bin': (2, 500),
            'num_leaves': (5, 500),
            'min_child_samples': (50, 1500),
            'min_child_weight': (0.1, 1000),
            'colsample_bytree': (0.05, 0.6),
            'reg_alpha': (0.1, 30),
            'reg_lambda': (0.1, 30),
        },
        probes=[
            {
                "colsample_bytree": 0.10334767151552683,
                "learning_rate": 0.029891107644305845,
                "max_bin": 403.63597687481536,
                "max_depth": 7.630901866600385,
                "min_child_samples": 867.6192673961293,
                "min_child_weight": 160.84184939114124,
                "num_leaves": 407.630333559809,
                "reg_alpha": 15.807080783827528,
                "reg_lambda": 10.109721556430491,
            },
            {
                "colsample_bytree": 0.7479871340435861,
                "learning_rate": 0.16420226500237103,
                "max_bin": 478.9454982979397,
                "max_depth": 7.074141701427287,
                "min_child_samples": 160.65491022582302,
                "min_child_weight": 22.407785965651843,
                "num_leaves": 265.60671862673746,
                "reg_alpha": 29.848926681241593,
                "reg_lambda": 27.132113615450297,
            },
            {
                "colsample_bytree": 0.1248098023982449,
                "learning_rate": 0.10342583432650229,
                "max_bin": 187.14525960895386,
                "max_depth": 4.758365045540807,
                "min_child_samples": 235.51063646456035,
                "min_child_weight": 7.445893668578389,
                "num_leaves": 193.04674974320685,
                "reg_alpha": 12.398553150143126,
                "reg_lambda": 24.496136535125633,
            }
        ]
    )

    bayes_opt(
        lgb_uncorrect_attempts_eval,
        {
            'learning_rate': (0.01, 1.0),
            'max_depth': (3, 10),
            'max_bin': (2, 500),
            'num_leaves': (5, 500),
            'min_child_samples': (50, 1500),
            'min_child_weight': (0.1, 1000),
            'colsample_bytree': (0.05, 0.6),
            'reg_alpha': (0.1, 30),
            'reg_lambda': (0.1, 30),
        },
        probes=[
            {
                "colsample_bytree": 0.3885796025348852,
                "learning_rate": 0.09146678093888543,
                "max_bin": 298.03316385733507,
                "max_depth": 7.439908933764865,
                "min_child_samples": 274.88036337243125,
                "min_child_weight": 361.1130703470717,
                "num_leaves": 317.361765580482,
                "reg_alpha": 5.2603473767855515,
                "reg_lambda": 28.75196606940299
            },
            {
                "colsample_bytree": 0.47037170385250415,
                "learning_rate": 0.012162585925199094,
                "max_bin": 25.980553497754865,
                "max_depth": 5.786974861042519,
                "min_child_samples": 1463.1282815050479,
                "min_child_weight": 131.64426437899147,
                "num_leaves": 471.97671065831497,
                "reg_alpha": 12.371956179828944,
                "reg_lambda": 19.2948688659447,
            },
            {
                "colsample_bytree": 0.3903008048793679,
                "learning_rate": 0.15882978384614804,
                "max_bin": 351.9710071636617,
                "max_depth": 9.379019489701598,
                "min_child_samples": 1304.882469005186,
                "min_child_weight": 498.0164877147689,
                "num_leaves": 16.041409064329585,
                "reg_alpha": 2.5843001310082196,
                "reg_lambda": 18.35238370653897,
            }
        ]
    )

    bayes_opt(
        lgb_accuracy_group_eval,
        {
            'learning_rate': (0.01, 1.0),
            'max_depth': (3, 10),
            'max_bin': (2, 500),
            'num_leaves': (5, 500),
            'min_child_samples': (50, 1500),
            'min_child_weight': (0.1, 1000),
            'colsample_bytree': (0.05, 0.6),
            'reg_alpha': (0.1, 30),
            'reg_lambda': (0.1, 30),
        }
    )
