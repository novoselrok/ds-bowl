import os

import numpy as np
from bayes_opt import BayesianOptimization, JSONLogger, Events
from bayes_opt.util import load_logs
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from submission import categorical_features, get_train_test_features, stratified_group_k_fold


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
            early_stopping_rounds=50,
            eval_set=[(x_val, y_val)],
            verbose=0,
            feature_name=feature_names,
            categorical_feature=categorical_features
        )

        pred = model.predict_proba(x_val).argmax(axis=1)
        return cohen_kappa_score(y_val, pred, weights='quadratic')

    for param in ['max_depth', 'max_bin', 'num_leaves', 'min_child_samples', 'n_splits', 'subsample_freq']:
        if param in params:
            params[param] = int(params[param])

    return cv(lgb_score, **params)


def catboost_eval(**params):
    def catboost_score(X_fold_train, X_fold_val, y_fold_train, y_fold_val, X_test, y_test, **model_params):
        model = CatBoostClassifier(
            iterations=5000,
            random_state=2019,
            use_best_model=True,
            loss_function='MultiClass',
            **model_params
        )

        model.fit(
            X_fold_train, y_fold_train,
            cat_features=categorical_feature_col_idxs,
            early_stopping_rounds=500,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=0,
        )

        pred = model.predict_proba(X_test).argmax(axis=1)
        return cohen_kappa_score(y_test, pred, weights='quadratic')

    for param in ['max_depth', 'n_splits']:
        if param in params:
            params[param] = int(params[param])

    return cv(catboost_score, **params)


def rfc_eval(**params):
    def rfc_score(X_fold_train, X_fold_val, y_fold_train, y_fold_val, X_test, y_test, **model_params):
        X_fold_train = X_fold_train.values
        X_fold_val = X_fold_val.values
        X_test = X_test.values

        ohe = OneHotEncoder(sparse=False)
        X_fold_train_categorical_encoded = ohe.fit_transform(X_fold_train[:, categorical_feature_col_idxs])
        X_fold_val_categorical_encoded = ohe.transform(X_fold_val[:, categorical_feature_col_idxs])
        X_test_categorical_encoded = ohe.transform(X_test[:, categorical_feature_col_idxs])

        X_fold_train = np.concatenate((X_fold_train[:, numerical_feature_col_idxs], X_fold_train_categorical_encoded), axis=1)
        X_fold_val = np.concatenate((X_fold_val[:, numerical_feature_col_idxs], X_fold_val_categorical_encoded), axis=1)
        X_test = np.concatenate((X_test[:, numerical_feature_col_idxs], X_test_categorical_encoded), axis=1)

        model = RandomForestClassifier(
            n_jobs=-1,
            random_state=2019,
            **model_params
        )

        model.fit(X_fold_train, y_fold_train)
        pred = model.predict_proba(np.concatenate((X_fold_val, X_test))).argmax(axis=1)
        return cohen_kappa_score(np.concatenate((y_fold_val, y_test)), pred, weights='quadratic')

    for param in ['n_splits', 'max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf']:
        if param in params:
            params[param] = int(params[param])

    return cv(rfc_score, **params)


def cv(score_fn, **params):
    n_splits = params['n_splits']
    del params['n_splits']

    scores = []
    for fold, (train_split, test_split) in enumerate(
            stratified_group_k_fold(X, y_target, train_installation_ids, n_splits, seed=2019)
    ):
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

    opt.maximize(n_iter=60, init_points=20)
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
    #     rfc_eval,
    #     {
    #         'n_splits': (5, 11),
    #         'n_estimators': (500, 2000),
    #         'max_depth': (3, 8),
    #         'max_features': (0.1, 1.0),
    #         'min_samples_split': (2, 255),
    #         'min_samples_leaf': (1, 100)
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
            'max_depth': (3, 6),
            'max_bin': (2, 128),
            'num_leaves': (5, 60),
            'min_child_samples': (50, 1000),
            'min_child_weight': (0.1, 1000),
            'subsample': (0.1, 1.0),
            'subsample_freq': (1, 10),
            'colsample_bytree': (0.1, 0.5),
            'reg_alpha': (1, 100),
            'reg_lambda': (1, 100)
        }
    )
