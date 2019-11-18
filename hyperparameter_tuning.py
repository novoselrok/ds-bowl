import os

import numpy as np
from bayes_opt import BayesianOptimization, JSONLogger, Events
from bayes_opt.util import load_logs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from lightgbm import LGBMClassifier, LGBMRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from submission import stratified_group_k_fold, get_train_test_features, label_encode_categorical_features

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)


def lgb_multi_model_eval(**params):
    (
        df_train_features,
        df_test_features,
        y_correct,
        y_uncorrect,
        y_accuracy_group,
        train_installation_ids,
        _test_installation_ids
    ) = get_train_test_features()

    integer_params = ['max_depth', 'max_bin', 'num_leaves',
                      'min_child_samples', 'n_splits', 'subsample_freq', 'bagging_freq']
    df_train, _ = label_encode_categorical_features(df_train_features, df_test_features)

    model_names = ['accuracy_group', 'correct_attempts', 'uncorrect_attempts']
    models_specs = []

    for name in model_names:
        model_params = {param.split(':')[1]: value for param, value in params.items() if param.startswith(name)}

        for param, value in model_params.items():
            if param in integer_params:
                model_params[param] = int(round(value))

        models_specs.append({'name': name, 'params': model_params})

    return cv(df_train, y_accuracy_group, y_correct, y_uncorrect, train_installation_ids, models_specs)


def get_model(model_spec):
    if model_spec['name'] == 'accuracy_group':
        return LGBMClassifier(
            random_state=2019,
            n_estimators=5000,
            n_jobs=-1,
            objective='multiclass',
            metric='multi_logloss',
            **model_spec['params']
        )
    elif model_spec['name'] == 'correct_attempts':
        return LGBMClassifier(
            random_state=2019,
            n_estimators=5000,
            n_jobs=-1,
            objective='binary',
            metric='binary_logloss',
            **model_spec['params']
        )
    elif model_spec['name'] == 'uncorrect_attempts':
        return LGBMRegressor(
            random_state=2019,
            n_estimators=5000,
            n_jobs=-1,
            **model_spec['params']
        )

    raise Exception(f'Unknown model {model_spec["name"]}')


def fit_model(
        model_spec,
        model,
        X_train, y_train_accuracy_group, y_train_correct_attempts, y_train_uncorrect_attempts,
        X_val, y_val_accuracy_group, y_val_correct_attempts, y_val_uncorrect_attempts
):
    if model_spec['name'] == 'accuracy_group':
        model.fit(
            X_train, y_train_accuracy_group,
            early_stopping_rounds=10,
            eval_set=[(X_val, y_val_accuracy_group)],
            verbose=0,
        )
    elif model_spec['name'] == 'correct_attempts':
        model.fit(
            X_train, y_train_correct_attempts,
            early_stopping_rounds=10,
            eval_set=[(X_val, y_val_correct_attempts)],
            verbose=0,
        )
    elif model_spec['name'] == 'uncorrect_attempts':
        model.fit(
            X_train, y_train_uncorrect_attempts,
            early_stopping_rounds=10,
            eval_set=[(X_val, y_val_uncorrect_attempts)],
            verbose=0,
        )
    else:
        raise Exception(f'Unknown model {model_spec["name"]}')


def predict_model(model_spec, model, X_test):
    if model_spec['name'] == 'accuracy_group':
        return model.predict_proba(X_test)
    elif model_spec['name'] == 'correct_attempts':
        return model.predict_proba(X_test)
    elif model_spec['name'] == 'uncorrect_attempts':
        return model.predict(X_test).reshape((-1, 1))

    raise Exception(f'Unknown model {model_spec["name"]}')


def get_train_test_split(df_data, targets, train_split, test_split):
    df_train = df_data.iloc[train_split, :]
    df_test = df_data.iloc[test_split, :]

    train_targets = [target[train_split] for target in targets]
    test_targets = [target[test_split] for target in targets]

    return tuple([df_train] + train_targets), tuple([df_test] + test_targets)


def oof_predict(
        X_train, y_train_accuracy_group, y_train_correct_attempts, y_train_uncorrect_attempts,
        X_test, y_test_accuracy_group, y_test_correct_attempts, y_test_uncorrect_attempts,
        X_holdout, models_specs
):
    predictions = []
    holdout_predictions = []

    for idx, model_spec in enumerate(models_specs):
        model = get_model(model_spec)

        fit_model(
            model_spec, model,
            X_train, y_train_accuracy_group, y_train_correct_attempts, y_train_uncorrect_attempts,
            X_test, y_test_accuracy_group, y_test_correct_attempts, y_test_uncorrect_attempts
        )

        predictions.append(predict_model(model_spec, model, X_test))
        holdout_predictions.append(predict_model(model_spec, model, X_holdout))

    return np.concatenate(predictions, axis=1), np.concatenate(holdout_predictions, axis=1)


def predict(df_data, target_accuracy_group, target_correct_attempts, target_uncorrect_attempts,
            train_split, test_split, X_holdout, models_specs):
    (X_train, y_train_accuracy_group, y_train_correct_attempts, y_train_uncorrect_attempts), \
        (X_test, y_test_accuracy_group, y_test_correct_attempts, y_test_uncorrect_attempts) = get_train_test_split(
        df_data, [target_accuracy_group, target_correct_attempts, target_uncorrect_attempts], train_split, test_split)

    return oof_predict(
        X_train, y_train_accuracy_group, y_train_correct_attempts, y_train_uncorrect_attempts,
        X_test, y_test_accuracy_group, y_test_correct_attempts, y_test_uncorrect_attempts,
        X_holdout, models_specs
    )


def cv(df_train, target_accuracy_group, target_correct_attempts, target_uncorrect_attempts, installation_ids,
       models_specs):
    n_outer_splits = 5
    n_inner_splits = 4

    oof_predictions = np.zeros(df_train.shape[0])

    outer_splits = stratified_group_k_fold(None, target_accuracy_group, installation_ids, n_outer_splits, seed=2019)
    for (train_split, test_split) in outer_splits:
        X_train = df_train.iloc[train_split, :]
        X_test = df_train.iloc[test_split, :]

        train_split_accuracy_group = target_accuracy_group[train_split]
        train_split_correct_attempts = target_correct_attempts[train_split]
        train_split_uncorrect_attempts = target_uncorrect_attempts[train_split]
        train_split_installation_ids = installation_ids.iloc[train_split]

        test_split_accuracy_group = target_accuracy_group[test_split]

        oof_train_predictions = np.zeros((X_train.shape[0], 4 + 2 + 1))
        test_prediction = np.zeros((X_test.shape[0], 4 + 2 + 1))

        inner_splits = stratified_group_k_fold(
            None, train_split_accuracy_group, train_split_installation_ids, n_inner_splits, seed=2019
        )
        for (inner_train_split, inner_test_split) in inner_splits:
            inner_test_split_prediction, holdout_prediction = \
                predict(X_train, train_split_accuracy_group, train_split_correct_attempts,
                        train_split_uncorrect_attempts, inner_train_split, inner_test_split, X_test, models_specs)

            oof_train_predictions[inner_test_split, :] = inner_test_split_prediction
            test_prediction += (holdout_prediction / n_inner_splits)

        ss = StandardScaler()
        meta_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', n_jobs=-1, max_iter=300)
        meta_clf.fit(ss.fit_transform(oof_train_predictions), train_split_accuracy_group)
        meta_prediction = meta_clf.predict(ss.transform(test_prediction))
        oof_predictions[test_split] = meta_prediction

        score = cohen_kappa_score(test_split_accuracy_group, meta_prediction, weights='quadratic')
        print('Done fold. Score:', score)

    score = cohen_kappa_score(target_accuracy_group, oof_predictions, weights='quadratic')
    print(score)
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

    opt.maximize(n_iter=200, init_points=60)
    print(opt.max)


if __name__ == '__main__':
    os.makedirs('bayes_opt_logs', exist_ok=True)

    bayes_opt(
        lgb_multi_model_eval,
        {
            'accuracy_group:learning_rate': (0.01, 1.0),
            'accuracy_group:max_depth': (3, 8),
            'accuracy_group:max_bin': (2, 255),
            'accuracy_group:num_leaves': (5, 500),
            'accuracy_group:min_child_samples': (50, 1500),
            'accuracy_group:min_child_weight': (0.1, 1000),
            'accuracy_group:colsample_bytree': (0.1, 0.6),
            'accuracy_group:reg_alpha': (0.1, 30),
            'accuracy_group:reg_lambda': (0.1, 30),

            'correct_attempts:learning_rate': (0.01, 1.0),
            'correct_attempts:max_depth': (3, 8),
            'correct_attempts:max_bin': (2, 255),
            'correct_attempts:num_leaves': (5, 500),
            'correct_attempts:min_child_samples': (50, 1500),
            'correct_attempts:min_child_weight': (0.1, 1000),
            'correct_attempts:colsample_bytree': (0.1, 0.6),
            'correct_attempts:reg_alpha': (0.1, 30),
            'correct_attempts:reg_lambda': (0.1, 30),

            'uncorrect_attempts:learning_rate': (0.01, 1.0),
            'uncorrect_attempts:max_depth': (3, 8),
            'uncorrect_attempts:max_bin': (2, 255),
            'uncorrect_attempts:num_leaves': (5, 500),
            'uncorrect_attempts:min_child_samples': (50, 1500),
            'uncorrect_attempts:min_child_weight': (0.1, 1000),
            'uncorrect_attempts:colsample_bytree': (0.1, 0.6),
            'uncorrect_attempts:reg_alpha': (0.1, 30),
            'uncorrect_attempts:reg_lambda': (0.1, 30),
        }
    )
