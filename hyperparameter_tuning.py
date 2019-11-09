import os

import numpy as np
from bayes_opt import BayesianOptimization, JSONLogger, Events
from bayes_opt.util import load_logs
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from submission import categorical_features, get_train_test_features


def lgb_eval(**params):
    def lgb_score(x_train, x_val, y_train, y_val, **model_params):
        x_train = x_train.values
        x_val = x_val.values

        model = LGBMClassifier(
            random_state=2019,
            n_estimators=5000,
            num_classes=4,
            objective='multiclass',
            metric='multi_logloss',
            n_jobs=-1,
            **model_params
        )

        model.fit(
            x_train, y_train,
            early_stopping_rounds=500,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            verbose=0,
            feature_name=feature_names,
            categorical_feature=categorical_features
        )

        pred = model.predict_proba(x_val).argmax(axis=1)
        return cohen_kappa_score(y_val, pred, weights='quadratic')

    for param in ['max_depth', 'num_leaves', 'min_child_samples', 'n_splits']:
        if param in params:
            params[param] = int(params[param])

    return cv(lgb_score, **params)


def catboost_eval(**params):
    def catboost_score(x_train, x_val, y_train, y_val, **model_params):
        model = CatBoostClassifier(
            iterations=5000,
            random_state=2019,
            use_best_model=True,
            loss_function='MultiClass',
            **model_params
        )

        model.fit(
            x_train, y_train,
            cat_features=categorical_feature_col_idxs,
            early_stopping_rounds=500,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            verbose=100,
        )

        pred = model.predict_proba(x_val).argmax(axis=1)
        return cohen_kappa_score(y_val, pred, weights='quadratic')

    for param in ['max_depth', 'n_splits']:
        if param in params:
            params[param] = int(params[param])

    return cv(catboost_score, **params)


def sgd_eval(**params):
    def sgd_score(x_train, x_val, y_train, y_val, **model_params):
        x_train = x_train.values
        x_val = x_val.values

        ss = StandardScaler()
        x_train[:, numerical_feature_col_idxs] = ss.fit_transform(x_train[:, numerical_feature_col_idxs])
        x_val[:, numerical_feature_col_idxs] = ss.transform(x_val[:, numerical_feature_col_idxs])

        model = SGDClassifier(
            loss='log',
            penalty='l2',
            n_jobs=-1,
            early_stopping=True,
            verbose=False,
            max_iter=5000,
            n_iter_no_change=10,
            **model_params
        )

        model.fit(x_train, y_train)
        pred = model.predict_proba(x_val).argmax(axis=1)
        return cohen_kappa_score(y_val, pred, weights='quadratic')

    for param in ['max_depth', 'n_splits']:
        if param in params:
            params[param] = int(params[param])

    return cv(sgd_score, **params)


def cv(score_fn, **params):
    kf = GroupKFold(n_splits=params['n_splits'])
    del params['n_splits']

    scores = []
    for fold, (train_split, test_split) in enumerate(kf.split(X, y_target, train_installation_ids)):
        x_train, x_val, y_train, y_val = X.iloc[train_split, :], X.iloc[test_split, :], \
                                         y_target[train_split], y_target[test_split]
        score = score_fn(x_train, x_val, y_train, y_val, **params)
        print(score)
        scores.append(score)

    return np.mean(scores)


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

    opt.maximize(n_iter=60, init_points=10)
    print(opt.max)


if __name__ == '__main__':
    os.makedirs('bayes_opt_logs', exist_ok=True)

    (
        df_train_features,
        df_test_features,
        y_target,
        train_installation_ids,
        _test_installation_ids
    ) = get_train_test_features()

    feature_names = df_train_features.columns.tolist()
    numerical_feature_col_idxs = [idx for idx, name in enumerate(feature_names) if name not in categorical_features]
    categorical_feature_col_idxs = [idx for idx, name in enumerate(feature_names) if name in categorical_features]

    # Features matrix
    X = df_train_features

    # bayes_opt(
    #     sgd_eval,
    #     {
    #         'n_splits': (5, 11),
    #         'alpha': (0.0001, 10),
    #     }
    # )

    # bayes_opt(
    #     catboost_eval,
    #     {
    #         'n_splits': (5, 11),
    #         'learning_rate': (0.01, 0.1),
    #         'max_depth': (3, 8),
    #         'l2_leaf_reg': (0, 10),
    #         'bagging_temperature': (0, 1),
    #         'colsample_bylevel': (0.1, 1.0)
    #     }
    # )

    bayes_opt(
        lgb_eval,
        {
            'n_splits': (5, 11),
            'learning_rate': (0.01, 0.1),
            'max_depth': (3, 8),
            'num_leaves': (5, 60),
            'min_child_samples': (50, 1000),
            'min_child_weight': (0.1, 1000),
            'subsample': (0.1, 1.0),
            'colsample_bytree': (0.1, 1.0),
            'reg_alpha': (0, 10),
            'reg_lambda': (0, 10)
        },
        probes=[
            {
                'n_splits': 8,
                'colsample_bytree': 0.5551535508116036,
                'learning_rate': 0.01859880300849997,
                'max_depth': 6,
                'min_child_samples': 55,
                'min_child_weight': 27.418512657045937,
                'num_leaves': 6,
                'reg_alpha': 7.554123013819799,
                'reg_lambda': 4.4094812663177265,
                'subsample': 0.9493215701448805
            }
        ]
    )
