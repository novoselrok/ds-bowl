import os

import numpy as np
from bayes_opt import BayesianOptimization, JSONLogger, Events
from bayes_opt.util import load_logs
from lightgbm import LGBMClassifier
from sklearn.metrics import cohen_kappa_score
from submission import get_train_test_features, stratified_group_k_fold, \
    label_encode_categorical_features, cohen_kappa_lgb_metric


def lgb_eval(**params):
    def lgb_score(x_train, _x_val, y_train, _y_val, val_installation_ids, **model_params):
        scores = []
        for (val_split, test_split) in stratified_group_k_fold(None, _y_val, val_installation_ids, 2, seed=2019):
            x_val = _x_val.iloc[val_split]
            y_val = _y_val[val_split]
            x_test = _x_val.iloc[test_split]
            y_test = _y_val[test_split]

            model = LGBMClassifier(
                random_state=2019,
                n_estimators=5000,
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
                eval_metric=cohen_kappa_lgb_metric
            )

            pred = model.predict_proba(x_test).argmax(axis=1)
            scores.append(cohen_kappa_score(y_test, pred, weights='quadratic'))

        print(scores)
        return np.mean(scores)

    (
        df_train_features,
        df_test_features,
        y_target,
        train_installation_ids,
        _test_installation_ids
    ) = get_train_test_features()

    integer_params = ['max_depth', 'max_bin', 'num_leaves', 'min_child_samples', 'n_splits', 'subsample_freq']
    df_train, _ = label_encode_categorical_features(df_train_features, df_test_features)
    return cv(lgb_score, df_train, y_target, train_installation_ids, integer_params, **params)


def cv(score_fn, df_train, y_target, train_installation_ids, integer_params, **params):
    n_splits = 5

    for param in integer_params:
        if param in params:
            params[param] = int(params[param])

    scores = []
    kf = stratified_group_k_fold(df_train, y_target, train_installation_ids, n_splits, seed=2019)
    for fold, (train_split, test_split) in enumerate(kf):
        x_train, x_val, y_train, y_val = df_train.iloc[train_split, :], df_train.iloc[test_split, :], \
                                         y_target[train_split], y_target[test_split]

        score = score_fn(x_train, x_val, y_train, y_val, train_installation_ids[test_split], **params)
        scores.append(score)

        print(score)

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

    bayes_opt(
        lgb_eval,
        {
            'learning_rate': (0.01, 1.0),
            'max_depth': (3, 15),
            'max_bin': (2, 500),
            'num_leaves': (5, 100),
            'min_child_samples': (50, 1500),
            'min_child_weight': (0.1, 1000),
            'colsample_bytree': (0.1, 1.0),
            'reg_alpha': (0.1, 30),
            'reg_lambda': (0.1, 30),
        }
    )
