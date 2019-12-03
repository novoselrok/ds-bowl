import csv
import json
import math
import os
import gc
import random
from collections import defaultdict, Counter
from functools import partial

import pandas as pd
import numpy as np
import scipy.optimize
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor, plot_importance
from sklearn.metrics import cohen_kappa_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

BIRD_MEASURER_ASSESSMENT = 'Bird Measurer (Assessment)'
INTEGER_PARAMS = ['max_depth', 'max_bin', 'num_leaves',
                  'min_child_samples', 'n_splits', 'subsample_freq']

np.random.seed(0)


def fillna0(df):
    return df.fillna(0.0)


def preprocess_events():
    def _postprocessing(game_sessions):
        for idx in range(len(game_sessions)):
            event_codes = dict(game_sessions[idx]['event_codes'])
            event_ids = dict(game_sessions[idx]['event_ids'])
            game_time = np.max(game_sessions[idx]['game_times'])
            game_times_sorted = np.array(list(sorted(game_sessions[idx]['game_times'])))

            if game_times_sorted.shape[0] <= 1:
                game_time_mean_diff = 0.0
                game_time_std_diff = 0.0
            else:
                game_times_diff = game_times_sorted[1:] - game_times_sorted[:-1]
                game_time_mean_diff = np.mean(game_times_diff)
                game_time_std_diff = np.std(game_times_diff)

            del game_sessions[idx]['game_times']
            del game_sessions[idx]['event_codes']
            del game_sessions[idx]['event_ids']

            correct, uncorrect = game_sessions[idx]['correct_attempts'], game_sessions[idx]['uncorrect_attempts']
            if correct == 0 and uncorrect == 0:
                # game_sessions[idx]['accuracy_rate'] = 0.0
                pass
            else:
                game_sessions[idx]['accuracy_rate'] = correct / float(correct + uncorrect)

            if correct == 0 and uncorrect == 0:
                pass
            elif correct == 1 and uncorrect == 0:
                game_sessions[idx]['accuracy_group'] = 3
                game_sessions[idx]['accuracy_group_3'] = 3
            elif correct == 1 and uncorrect == 1:
                game_sessions[idx]['accuracy_group'] = 2
                game_sessions[idx]['accuracy_group_2'] = 1
            elif correct == 1 and uncorrect >= 2:
                game_sessions[idx]['accuracy_group'] = 1
                game_sessions[idx]['accuracy_group_1'] = 1
            else:
                game_sessions[idx]['accuracy_group'] = 0
                game_sessions[idx]['accuracy_group_0'] = 1

            game_sessions[idx] = {
                **game_sessions[idx],
                **event_codes,
                **event_ids,
                'game_time': game_time,
                'game_time_mean_diff': game_time_mean_diff,
                'game_time_std_diff': game_time_std_diff,
            }

        return game_sessions

    def _aggregate_game_sessions(csv_path, prefix, installation_ids_to_keep=None):
        game_sessions = {}
        with open(csv_path, encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)

            for idx, row in enumerate(reader):
                if installation_ids_to_keep and row['installation_id'] not in installation_ids_to_keep:
                    continue

                if row['game_session'] not in game_sessions:
                    game_sessions[row['game_session']] = {
                        'game_session': row['game_session'],
                        'timestamp': row['timestamp'],
                        'installation_id': row['installation_id'],
                        'event_count': 0,
                        'game_times': [],
                        'correct_attempts': 0,
                        'uncorrect_attempts': 0,
                        'title': row['title'],
                        'type': row['type'],
                        'world': row['world'],
                        'event_codes': defaultdict(int),
                        'event_ids': defaultdict(int),
                    }

                game_session = game_sessions[row['game_session']]

                game_session['event_count'] = max(game_session['event_count'], int(row['event_count']))
                game_session['game_times'].append(int(row['game_time']))

                event_data = json.loads(row['event_data'])
                event_code = int(row['event_code'])

                game_session['event_codes'][f'event_code_{event_code}'] += 1

                is_assessment = row['type'] == 'Assessment'
                if is_assessment:
                    is_bird_measurer_attempt = row['title'] == BIRD_MEASURER_ASSESSMENT and event_code == 4110
                    is_non_bird_measurer_attempt = row['title'] != BIRD_MEASURER_ASSESSMENT and event_code == 4100
                    is_assessment_attempt = is_bird_measurer_attempt or is_non_bird_measurer_attempt

                    if is_assessment_attempt:
                        is_correct_attempt = 'correct' in event_data and event_data['correct']
                        is_uncorrect_attempt = 'correct' in event_data and not event_data['correct']

                        if is_correct_attempt and is_assessment_attempt:
                            game_session['correct_attempts'] += 1

                        if is_uncorrect_attempt and is_assessment_attempt:
                            game_session['uncorrect_attempts'] += 1

                event_id = row['event_id']
                game_session['event_ids'][f'event_id_{event_id}'] += 1

                if idx % 10000 == 0:
                    print(idx)

        df_data = pd.DataFrame(_postprocessing(list(game_sessions.values())))
        fillna0_columns = [column for column in df_data.columns if not column.startswith('accuracy')]
        df_data[fillna0_columns] = df_data[fillna0_columns].fillna(0.0)
        df_data.to_csv(f'preprocessed-data/{prefix}_game_sessions.csv', index=False)

    os.makedirs('preprocessed-data', exist_ok=True)

    if not os.path.exists(TRAIN_FEATURES_CSV):
        df_train_labels = pd.read_csv(TRAIN_LABELS_CSV)

        # Only keep installation ids from train labels
        _aggregate_game_sessions(
            TRAIN_CSV,
            'train',
            installation_ids_to_keep=set(df_train_labels['installation_id'].unique())
        )

    _aggregate_game_sessions(
        TEST_CSV,
        'test'
    )

    pd.read_csv(TEST_CSV) \
        .groupby(['installation_id']) \
        .last() \
        .reset_index() \
        .to_csv('preprocessed-data/test_assessments_to_predict.csv', index=False)


def feature_engineering():
    aggfns = {'std': np.nanstd, 'mean': np.nanmean, 'max': np.nanmax}

    def _add_time_features(df):
        timestamp = pd.to_datetime(df['timestamp'])
        df['timestamp'] = (timestamp.astype(int) / 10 ** 9).astype(int)
        df['assessment_hour'] = timestamp.dt.hour
        df['assessment_dayofweek'] = timestamp.dt.dayofweek

    def _agg_game_sessions(values, columns, prefix=''):
        agg_columns = []
        agg = []
        for name, aggfn in aggfns.items():
            agg_columns.extend(
                [f'{prefix}{column}_{name}' for column in columns]
            )
            agg.append(
                aggfn(values, axis=0).reshape(1, -1)
            )

        return pd.DataFrame(np.concatenate(agg, axis=1), columns=agg_columns)

    def _compute_features(df_data, prefix, df_assessments):
        game_sessions = []

        event_codes_columns = [column for column in df_data.columns if column.startswith('event_code')]
        event_ids_columns = [column for column in df_data.columns if column.startswith('event_id')]

        df_assessments = df_assessments.sort_values(by='installation_id')
        installation_id_to_game_sessions = dict(tuple(df_data.groupby('installation_id')))
        title_to_game_sessions = dict(tuple(df_data.groupby('title')))

        _aggregate_game_sessions_columns = (aggregate_game_sessions_columns +
                                            event_codes_columns +
                                            event_ids_columns)

        total_assessments = df_assessments.shape[0]
        for idx, assessment in df_assessments.iterrows():
            installation_id = assessment['installation_id']
            start_timestamp = assessment['timestamp']

            game_sessions_for_installation_id = installation_id_to_game_sessions[installation_id]

            previous_game_sessions: pd.DataFrame = game_sessions_for_installation_id[
                game_sessions_for_installation_id['timestamp'] < start_timestamp
                ].copy(deep=True)

            game_sessions_for_title = title_to_game_sessions[assessment['title']]
            previous_global_assesment_game_sessions = game_sessions_for_title[
                (game_sessions_for_title['timestamp'] < start_timestamp) &
                (game_sessions_for_title['installation_id'] != installation_id)
                ][aggregate_assessment_game_session_columns].values

            if previous_global_assesment_game_sessions.shape[0] > 0:
                previous_global_assesment_game_sessions_agg = _agg_game_sessions(
                    previous_global_assesment_game_sessions, aggregate_assessment_game_session_columns,
                    prefix='previous_global_title_attempt_'
                )
            else:
                previous_global_assesment_game_sessions_agg = pd.DataFrame()

            assessment_info = pd.DataFrame({
                'installation_id': installation_id,
                'assessment_game_session': assessment['game_session'],
                'assessment_title': assessment['title'],
                'assessment_world': assessment['world'],
                'assessment_most_common_title_accuracy_group': assessment[
                    'assessment_most_common_title_accuracy_group'],
                'assessment_dayofweek': assessment['assessment_dayofweek'],
                'assessment_hour': assessment['assessment_hour'],
            }, index=[0])

            if previous_game_sessions.shape[0] == 0:
                game_sessions.append(pd.concat((assessment_info, previous_global_assesment_game_sessions_agg), axis=1))
                continue

            # Previous attempts for current assessment
            # Which attempt, accuracy groups for current assessment, correct/incorrect/rate attempts
            previous_user_assessment_game_sessions = previous_game_sessions[
                previous_game_sessions['title'] == assessment['title']
                ][aggregate_assessment_game_session_columns].values

            if previous_user_assessment_game_sessions.shape[0] > 0:
                previous_user_assessment_game_sessions_agg = _agg_game_sessions(
                    previous_user_assessment_game_sessions, aggregate_assessment_game_session_columns,
                    prefix='previous_user_title_attempt_'
                )
            else:
                previous_user_assessment_game_sessions_agg = pd.DataFrame()

            # One-hot-encode categorical features
            previous_game_sessions = pd.get_dummies(
                previous_game_sessions, columns=['title', 'type', 'world'], prefix='ohe')
            ohe_columns = [column for column in previous_game_sessions.columns if column.startswith('ohe')]
            ohe_agg = pd.DataFrame(
                np.nansum(previous_game_sessions[ohe_columns].values, axis=0).reshape(1, -1), columns=ohe_columns)

            previous_game_sessions_values = previous_game_sessions[_aggregate_game_sessions_columns].values
            previous_game_sessions_agg = _agg_game_sessions(
                previous_game_sessions_values, _aggregate_game_sessions_columns)

            df_final_agg = pd.concat(
                (assessment_info, previous_game_sessions_agg, ohe_agg, previous_user_assessment_game_sessions_agg,
                 previous_global_assesment_game_sessions_agg),
                axis=1
            )

            game_sessions.append(df_final_agg)

            # if idx == 10:
            #     break

            if idx % 100 == 0:
                print(f'Row {idx + 1}/{total_assessments} done')

        df_final = pd.concat(game_sessions, ignore_index=True, sort=False)
        print('Writing features...')
        df_final.to_csv(f'preprocessed-data/{prefix}_features.csv', index=False)

    aggregate_game_sessions_columns = [
        'game_time',
        'event_count',
        'game_time_mean_diff',
        'game_time_std_diff',
    ]

    aggregate_assessment_game_session_columns = aggregate_game_sessions_columns + [
        'correct_attempts',
        'uncorrect_attempts',
        'accuracy_rate',
        'accuracy_group',
        'accuracy_group_3',
        'accuracy_group_2',
        'accuracy_group_1',
        'accuracy_group_0',
    ]

    df_train_labels = pd.read_csv(TRAIN_LABELS_CSV)
    # Add most common title accuracy group to assessment
    title_to_acc_group_dict = dict(
        df_train_labels.groupby('title')['accuracy_group'].agg(lambda x: x.value_counts().index[0])
    )

    if not os.path.exists(TRAIN_FEATURES_CSV):
        print('Preparing train data...')
        df_train = pd.read_csv('preprocessed-data/train_game_sessions.csv')

        title_to_world = df_train[['title', 'world']].groupby('title').agg('first').apply(list).to_dict()
        # Add assessment world
        df_train_labels['world'] = df_train_labels.apply(
            lambda assessment: title_to_world['world'][assessment['title']],
            axis=1
        )

        df_train_labels['assessment_most_common_title_accuracy_group'] = df_train_labels['title'] \
            .map(title_to_acc_group_dict)

        # Add game session start timestamp to train labels
        game_session_start_timestamps = df_train.groupby(
            ['installation_id', 'game_session']
        ).first()['timestamp'].reset_index()

        df_train_labels = df_train_labels.merge(
            game_session_start_timestamps, on=['installation_id', 'game_session'], how='left'
        )

        _add_time_features(df_train_labels)
        df_train['timestamp'] = (pd.to_datetime(df_train['timestamp']).astype(int) / 10 ** 9).astype(int)

        _compute_features(df_train, 'train', df_train_labels)

        del df_train
        gc.collect()

    print('Preparing test data...')
    df_test = pd.read_csv('preprocessed-data/test_game_sessions.csv')
    df_test_assessments_to_predict = pd.read_csv('preprocessed-data/test_assessments_to_predict.csv')
    df_test_assessments_to_predict['assessment_most_common_title_accuracy_group'] = \
        df_test_assessments_to_predict['title'].map(title_to_acc_group_dict)

    _add_time_features(df_test_assessments_to_predict)
    df_test['timestamp'] = (pd.to_datetime(df_test['timestamp']).astype(int) / 10 ** 9).astype(int)

    _compute_features(
        df_test,
        'test',
        df_test_assessments_to_predict
    )

    del df_test
    gc.collect()


def label_encode_categorical_features(df_train, df_test=None):
    for column in categorical_features:
        lb = LabelEncoder()
        train_column = df_train[column]
        lb.fit(train_column)
        df_train[column] = lb.transform(train_column)
        df_train[column] = df_train[column].astype('category')

        if df_test is not None:
            test_column = df_test[column]
            df_test[column] = lb.transform(test_column)
            df_test[column] = df_test[column].astype('category')

    return df_train, df_test


def ohe_encode_categorical_features(df_train, df_test=None):
    for column in categorical_features:
        lb = LabelEncoder()
        train_column = df_train[column]
        lb.fit(train_column)
        df_train[column] = lb.transform(train_column)
        df_train = pd.get_dummies(df_train, columns=[column])

        if df_test is not None:
            test_column = df_test[column]
            df_test[column] = lb.transform(test_column)
            df_test = pd.get_dummies(df_test, columns=[column])

    return df_train, df_test


def get_train_test_features():
    columns_to_drop = [
        'installation_id',
        'assessment_game_session',
    ]

    df_train_features = pd.read_csv(TRAIN_FEATURES_CSV)
    df_test_features = pd.read_csv('preprocessed-data/test_features.csv')

    df_train_labels = pd.read_csv(
        TRAIN_LABELS_CSV,
        usecols=['installation_id', 'game_session', 'num_correct', 'num_incorrect', 'accuracy', 'accuracy_group']
    ).rename({'game_session': 'assessment_game_session'}, axis=1)

    # Add target to train features
    df_train_features = df_train_features.merge(df_train_labels, on=['installation_id', 'assessment_game_session'])

    # Extract target and installation ids
    y_correct = df_train_features['num_correct'].values
    y_uncorrect = df_train_features['num_incorrect'].values
    y_accuracy_rate = df_train_features['accuracy'].values
    y_accuracy_group = df_train_features['accuracy_group'].values
    train_installation_ids = df_train_features['installation_id']
    test_installation_ids = df_test_features['installation_id']

    df_train_features = df_train_features.drop(
        columns=columns_to_drop + ['num_correct', 'num_incorrect', 'accuracy', 'accuracy_group']
    )
    df_train_features_columns = set(df_train_features.columns)
    df_test_features_columns = set(df_test_features.columns)
    columns = df_train_features_columns & df_test_features_columns

    # Make sure test columns are in the correct order
    df_train_features = df_train_features[columns]
    df_test_features = df_test_features.drop(columns=columns_to_drop)[columns]

    return {
        'df_train_features': df_train_features,
        'df_test_features': df_test_features,
        'y_correct': y_correct,
        'y_uncorrect': y_uncorrect,
        'y_accuracy_rate': y_accuracy_rate,
        'y_accuracy_group': y_accuracy_group,
        'train_installation_ids': train_installation_ids,
        'test_installation_ids': test_installation_ids
    }


def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def get_lgbm_classifier(objective, metric, n_estimators=5000):
    def inner(params):
        return LGBMClassifier(
            random_state=2019,
            n_estimators=n_estimators,
            n_jobs=-1,
            objective=objective,
            metric=metric,
            subsample_freq=1,
            **params
        )

    return inner


def get_lgbm_regressor(objective, metric, n_estimators=5000):
    def inner(params):
        return LGBMRegressor(
            random_state=2019,
            n_estimators=n_estimators,
            n_jobs=-1,
            objective=objective,
            metric=metric,
            subsample_freq=1,
            **params
        )

    return inner


def get_catboost_classifier(objective, metric, cat_features, n_estimators=5000):
    def inner(params):
        return CatBoostClassifier(
            random_state=2019,
            loss_function=objective,
            eval_metric=metric,
            n_estimators=n_estimators,
            cat_features=cat_features,
            use_best_model=True,
            **params
        )

    return inner


def get_catboost_regressor(objective, metric, cat_features, n_estimators=5000):
    def inner(params):
        return CatBoostRegressor(
            random_state=2019,
            loss_function=objective,
            eval_metric=metric,
            n_estimators=n_estimators,
            cat_features=cat_features,
            use_best_model=True,
            **params
        )

    return inner


def integer_encode_params(params):
    for param, value in params.items():
        if param in INTEGER_PARAMS:
            params[param] = int(round(value))
    return params


def fit_model(model, X_train, y_train, X_val, y_val, early_stopping_rounds=100, verbose=0):
    model.fit(
        X_train, y_train,
        early_stopping_rounds=early_stopping_rounds,
        eval_set=[(X_val, y_val)],
        verbose=verbose,
    )


def cv_with_oof_predictions(
        model_fn, score_fn, X, X_test, y, y_accuracy_group, installation_ids, model_params,
        n_splits=5, n_predicted_features=2, predict_proba=True
):
    scores = []
    oof_train_predictions = np.zeros((X.shape[0], n_predicted_features))
    test_predictions = np.zeros((X_test.shape[0], n_predicted_features))

    kf = stratified_group_k_fold(None, y_accuracy_group, installation_ids, n_splits, seed=2019)
    for fold, (train_split, test_split) in enumerate(kf):
        X_train, X_val = X.iloc[train_split, :], X.iloc[test_split, :]
        y_train, y_val = y[train_split], y[test_split]

        model = model_fn(model_params)
        fit_model(model, X_train, y_train, X_val, y_val)

        if predict_proba:
            val_prediction = model.predict_proba(X_val)
            test_prediction = model.predict_proba(X_test)
        else:
            val_prediction = model.predict(X_val).reshape((-1, 1))
            test_prediction = model.predict(X_test).reshape((-1, 1))

        oof_train_predictions[test_split, :] = val_prediction
        test_predictions += (test_prediction / n_splits)

        scores.append(score_fn(y_val, model.predict(X_val)))

    print('CV score:', np.mean(scores))

    return oof_train_predictions, test_predictions


def fit_predict_correct_attempts_model(params):
    features = get_train_test_features()

    df_train, df_test = label_encode_categorical_features(features['df_train_features'], features['df_test_features'])
    model_params = integer_encode_params(params)

    meta_train, meta_test = cv_with_oof_predictions(
        get_lgbm_classifier('binary', 'auc'),
        roc_auc_score,
        df_train,
        df_test,
        features['y_correct'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params
    )

    return meta_train[:, 1].reshape(-1, 1), meta_test[:, 1].reshape(-1, 1)


def fit_predict_accuracy_group_regression_model(params):
    features = get_train_test_features()

    df_train, df_test = label_encode_categorical_features(features['df_train_features'], features['df_test_features'])
    model_params = integer_encode_params(params)

    return cv_with_oof_predictions(
        get_lgbm_regressor('regression', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        df_train,
        df_test,
        features['y_accuracy_group'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params,
        predict_proba=False,
        n_predicted_features=1
    )


def fit_predict_accuracy_rate_regression_model(params):
    features = get_train_test_features()

    df_train, df_test = label_encode_categorical_features(features['df_train_features'], features['df_test_features'])
    model_params = integer_encode_params(params)

    return cv_with_oof_predictions(
        get_lgbm_regressor('regression', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        df_train,
        df_test,
        features['y_accuracy_rate'],
        features['y_accuracy_group'],
        features['train_installation_ids'],
        model_params,
        predict_proba=False,
        n_predicted_features=1
    )


def binarize_accuracy_group(y_accuracy_group, target):
    pos_label_indices = y_accuracy_group > target
    neg_label_indices = y_accuracy_group <= target
    y_accuracy_group[pos_label_indices] = 1
    y_accuracy_group[neg_label_indices] = 0
    return y_accuracy_group


def predict_ordinal_accuracy_group(train_predictions, test_predictions):
    n_train = train_predictions[0].shape[0]
    n_test = test_predictions[0].shape[0]
    train_accuracy_group_proba = np.zeros((n_train, 4))
    test_accuracy_group_proba = np.zeros((n_test, 4))

    train_accuracy_group_proba[:, 0] = np.ones(n_train) - train_predictions[0]
    train_accuracy_group_proba[:, 1] = train_predictions[0] - train_predictions[1]
    train_accuracy_group_proba[:, 2] = train_predictions[1] - train_predictions[2]
    train_accuracy_group_proba[:, 3] = train_predictions[2]

    test_accuracy_group_proba[:, 0] = np.ones(n_test) - test_predictions[0]
    test_accuracy_group_proba[:, 1] = test_predictions[0] - test_predictions[1]
    test_accuracy_group_proba[:, 2] = test_predictions[1] - test_predictions[2]
    test_accuracy_group_proba[:, 3] = test_predictions[2]

    return (train_accuracy_group_proba.argmax(axis=1).reshape(-1, 1),
            test_accuracy_group_proba.argmax(axis=1).reshape(-1, 1))


def fit_predict_ordinal_model(params):
    train_predictions = []
    test_predictions = []
    for target in [0, 1, 2]:
        features = get_train_test_features()

        y_accuracy_group = binarize_accuracy_group(features['y_accuracy_group'], target)

        df_train, df_test = label_encode_categorical_features(features['df_train_features'],
                                                              features['df_test_features'])
        model_params = integer_encode_params(params[target])

        train_prediction, test_prediction = cv_with_oof_predictions(
            get_lgbm_classifier('binary', 'auc'),
            roc_auc_score,
            df_train,
            df_test,
            y_accuracy_group,
            y_accuracy_group,
            features['train_installation_ids'],
            model_params,
        )

        train_predictions.append(train_prediction[:, 1])
        test_predictions.append(test_prediction[:, 1])

    return predict_ordinal_accuracy_group(train_predictions, test_predictions)


class OptimizedRounder(object):
    def __init__(self, initial_coef, labels):
        self.coef_ = 0
        self.initial_coef = initial_coef
        self.labels = labels

    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=self.labels)
        return -cohen_kappa_score(y, preds, weights='quadratic')

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        self.coef_ = scipy.optimize.minimize(loss_partial, self.initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=self.labels)
        return preds

    def coefficients(self):
        return self.coef_['x']


def output_submission():
    models = [
        {
            'name': 'correct_attempts',
            # 'thresholds': [0.4, 0.8, 0.95],
            'thresholds': [lambda: np.random.uniform(0.3, 0.5), lambda: np.random.uniform(0.5, 0.85), lambda: np.random.uniform(0.85, 0.95)],
            'params': {'colsample_bytree': 0.7119967076685797,
                       'learning_rate': 0.3898073223635955,
                       'max_bin': 490.7257788082861,
                       'max_depth': 6.393385888236407,
                       'min_child_samples': 618.8943108109222,
                       'min_child_weight': 242.39854189044027,
                       'num_leaves': 226.53292461740185,
                       'reg_alpha': 29.112808215086247,
                       'reg_lambda': 1.442864888247586,
                       'subsample': 0.9508616840652329}
        },
        {
            'name': 'accuracy_group_regression',
            # 'thresholds': [0.5, 1.5, 2.5],
            'thresholds': [lambda: np.random.uniform(0, 1), lambda: np.random.uniform(1, 2), lambda: np.random.uniform(2, 3)],
            'params': {'colsample_bytree': 0.10220450673829048,
                       'learning_rate': 0.018036788162107645,
                       'max_bin': 495.15682953018984,
                       'max_depth': 12.957323504526952,
                       'min_child_samples': 55.28291794257959,
                       'min_child_weight': 484.471987537883,
                       'num_leaves': 487.9146357293617,
                       'reg_alpha': 12.208540618209394,
                       'reg_lambda': 14.820117618503028,
                       'subsample': 0.9553498912171228}
        },
        {
            'name': 'accuracy_rate_regression',
            # 'thresholds': [0.25, 0.5, 0.75],
            'thresholds': [lambda: np.random.uniform(0.1, 0.3), lambda: np.random.uniform(0.4, 0.6), lambda: np.random.uniform(0.7, 0.9)],
            'params': {'colsample_bytree': 0.3219064135017099,
                       'learning_rate': 0.012346429656658994,
                       'max_bin': 13.58960894887651,
                       'max_depth': 3.9084457978764915,
                       'min_child_samples': 578.9230545329486,
                       'min_child_weight': 254.75521773098373,
                       'num_leaves': 208.59965593534642,
                       'reg_alpha': 20.973836559719214,
                       'reg_lambda': 28.4873802178074,
                       'subsample': 0.8351287838482833}
        },
        {
            'name': 'accuracy_group_ordinal',
            'params': [
                {'colsample_bytree': 0.39370372351569727,
                 'learning_rate': 0.2977323934075598,
                 'max_bin': 489.4294536649877,
                 'max_depth': 8.739684209308205,
                 'min_child_samples': 454.3209316311202,
                 'min_child_weight': 18.192270605066476,
                 'num_leaves': 494.896099744235,
                 'reg_alpha': 28.465109515910957,
                 'reg_lambda': 29.8788782909773,
                 'subsample': 0.8020994065767717},
                {'colsample_bytree': 0.8743807726819987,
                 'learning_rate': 0.0503587517143143,
                 'max_bin': 89.2179685244991,
                 'max_depth': 14.4251619048779,
                 'min_child_samples': 226.81637132874744,
                 'min_child_weight': 118.02167568264719,
                 'num_leaves': 10.419557797232418,
                 'reg_alpha': 3.4139768591952686,
                 'reg_lambda': 28.977438242669628,
                 'subsample': 0.7572929646845502},
                {'colsample_bytree': 0.711800015732685,
                 'learning_rate': 0.22948791016816208,
                 'max_bin': 113.43166482130843,
                 'max_depth': 15.83595141260532,
                 'min_child_samples': 859.5779870826568,
                 'min_child_weight': 598.4798664200113,
                 'num_leaves': 253.4183460048864,
                 'reg_alpha': 3.9659199358286896,
                 'reg_lambda': 29.771125171182824,
                 'subsample': 0.9999786229569586}
            ]
        },
    ]

    def output_meta_features():
        base_features = []
        for model_desc in models:
            if model_desc['name'] == 'correct_attempts':
                base_features.append(fit_predict_correct_attempts_model(model_desc['params']))
            elif model_desc['name'] == 'accuracy_group_regression':
                base_features.append(fit_predict_accuracy_group_regression_model(model_desc['params']))
            elif model_desc['name'] == 'accuracy_rate_regression':
                base_features.append(fit_predict_accuracy_rate_regression_model(model_desc['params']))
            elif model_desc['name'] == 'accuracy_group_ordinal':
                base_features.append(fit_predict_ordinal_model(model_desc['params']))

        meta_train_features = np.concatenate([_train for (_train, _) in base_features], axis=1)
        meta_test_features = np.concatenate([_test for (_, _test) in base_features], axis=1)

        train_test_features = get_train_test_features()
        y_accuracy_group = train_test_features['y_accuracy_group']
        train_installation_ids = train_test_features['train_installation_ids']
        test_installation_ids = train_test_features['test_installation_ids']

        columns = [model_desc['name'] for model_desc in models]
        df_meta_train = pd.DataFrame(meta_train_features, columns=columns)
        df_meta_train['target'] = y_accuracy_group
        df_meta_train['installation_ids'] = train_installation_ids

        df_meta_test = pd.DataFrame(meta_test_features, columns=columns)
        df_meta_test['installation_ids'] = test_installation_ids

        df_meta_train.to_csv('preprocessed-data/meta_train_features.csv', index=False)
        df_meta_test.to_csv('preprocessed-data/meta_test_features.csv', index=False)

    def output_test_predictions():
        df_meta_test_features = pd.read_csv('preprocessed-data/meta_test_features.csv')
        df_meta_train_features = pd.read_csv('preprocessed-data/meta_train_features.csv')
        target = df_meta_train_features['target'].values
        test_installation_ids = df_meta_test_features['installation_ids']

        best_score = 0.0
        best_test_prediction = None

        for _ in range(100):
            train_scores = {}
            meta_train_thresholded = {}
            meta_test_thresholded = {}

            for model_desc in models:
                if 'thresholds' in model_desc:
                    thresholds = [rand_fn() for rand_fn in model_desc['thresholds']]
                    or_ = OptimizedRounder(thresholds, [0, 1, 2, 3])
                    or_.fit(df_meta_train_features[model_desc['name']], target)

                    train_prediction = or_.predict(df_meta_train_features[model_desc['name']], or_.coefficients())
                    test_prediction = or_.predict(df_meta_test_features[model_desc['name']], or_.coefficients())

                    train_score = cohen_kappa_score(train_prediction, target, weights='quadratic')
                    print(
                        model_desc['name'],
                        or_.coefficients(),
                        train_score
                    )

                    meta_train_thresholded[model_desc['name']] = train_prediction.values
                    meta_test_thresholded[model_desc['name']] = test_prediction.values
                else:
                    meta_train_thresholded[model_desc['name']] = df_meta_train_features[
                        model_desc['name']].values.astype(int)
                    meta_test_thresholded[model_desc['name']] = df_meta_test_features[model_desc['name']].values.astype(int)

                    train_score = cohen_kappa_score(meta_train_thresholded[model_desc['name']], target, weights='quadratic')
                    print(
                        model_desc['name'],
                        train_score
                    )

                train_scores[model_desc['name']] = train_score

            df_train_prediction = pd.DataFrame(meta_train_thresholded)
            df_test_prediction = pd.DataFrame(meta_test_thresholded)[df_train_prediction.columns]

            train_weights = [train_scores[column] for column in df_train_prediction.columns]
            train_weights = np.array(train_weights) / np.sum(train_weights)

            train_prediction = []
            for _, row in df_train_prediction.iterrows():
                pred = np.bincount(row.values.astype(int), minlength=4, weights=train_weights).argmax()
                train_prediction.append(pred)

            test_prediction = []
            for _, row in df_test_prediction.iterrows():
                pred = np.bincount(row.values.astype(int), minlength=4, weights=train_weights).argmax()
                test_prediction.append(pred)

            score = cohen_kappa_score(train_prediction, target, weights='quadratic')
            print(score)

            if score > best_score:
                best_score = score
                best_test_prediction = test_prediction

        pd.DataFrame.from_dict({
            'installation_id': test_installation_ids,
            'accuracy_group': best_test_prediction
        }).to_csv('submission.csv', index=False)

    output_meta_features()
    output_test_predictions()


categorical_features = [
    'assessment_title',
    'assessment_world',
]

TRAIN_CSV = 'data/train.csv'
TRAIN_LABELS_CSV = 'data/train_labels.csv'
TEST_CSV = 'data/test.csv'
TRAIN_FEATURES_CSV = 'preprocessed-data/train_features.csv'
EVENT_PROPS_JSON = 'event_props.json'
CORRELATED_FEATURES_JSON = 'correlated_features.json'
FEATURE_IMPORTANCES_CSV = 'feature_importances.csv'
