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
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import cohen_kappa_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

important_event_ids = {'05ad839b',
                       '08ff79ad',
                       '0d18d96c',
                       '0db6d71d',
                       '1325467d',
                       '15a43e5b',
                       '19967db1',
                       '222660ff',
                       '2230fab4',
                       '250513af',
                       '262136f4',
                       '28a4eb9a',
                       '28f975ea',
                       '2b058fe3',
                       '2dc29e21',
                       '30614231',
                       '31973d56',
                       '3393b68b',
                       '363c86c9',
                       '37937459',
                       '3afb49e6',
                       '3afde5dd',
                       '3bb91dda',
                       '3bf1cf26',
                       '3d63345e',
                       '3ddc79c3',
                       '3edf6747',
                       '3ee399c3',
                       '45d01abe',
                       '461eace6',
                       '47026d5f',
                       '47efca07',
                       '47f43a44',
                       '499edb7c',
                       '4bb2f698',
                       '4e5fc6f5',
                       '4ef8cdd3',
                       '5290eab1',
                       '565a3990',
                       '5859dfb6',
                       '587b5989',
                       '5c2f29ca',
                       '5c3d2b2f',
                       '5e3ea25a',
                       '5f0eb72c',
                       '6043a2b4',
                       '67aa2ada',
                       '6aeafed4',
                       '6c517a88',
                       '6f4adc4b',
                       '6f4bd64e',
                       '6f8106d9',
                       '731c0cbe',
                       '7372e1a5',
                       '73757a5e',
                       '74e5f8a7',
                       '76babcde',
                       '77c76bc5',
                       '77ead60d',
                       '7da34a02',
                       '7dfe6d8a',
                       '804ee27f',
                       '84538528',
                       '85d1b0de',
                       '86c924c4',
                       '884228c8',
                       '88d4a5be',
                       '89aace00',
                       '8af75982',
                       '8b757ab8',
                       '8d748b58',
                       '8fee50e2',
                       '90efca10',
                       '92687c59',
                       '93edfe2e',
                       '9b4001e4',
                       '9ce586dd',
                       '9d29771f',
                       '9e34ea74',
                       'a0faea5d',
                       'a16a373e',
                       'a2df0760',
                       'a44b10dc',
                       'a5be6304',
                       'a7640a16',
                       'a8876db3',
                       'ac92046e',
                       'acf5c23f',
                       'b2e5b0f1',
                       'bc8f2793',
                       'bcceccc6',
                       'bd612267',
                       'c0415e5c',
                       'c51d8688',
                       'c58186bf',
                       'c7fe2a55',
                       'c952eb01',
                       'cb1178ad',
                       'cdd22e43',
                       'cf82af56',
                       'd02b7a8e',
                       'd185d3ea',
                       'd2e9262e',
                       'd3268efa',
                       'd3640339',
                       'd38c2fd7',
                       'd3f1e122',
                       'd45ed6a1',
                       'daac11b0',
                       'de26c3a6',
                       'df4fe8b6',
                       'e04fb33d',
                       'e080a381',
                       'e37a2b78',
                       'e3ff61fb',
                       'e57dd7af',
                       'e694a35b',
                       'e7e44842',
                       'eb2c19cd',
                       'f3cd5473',
                       'f54238ee'}

BIRD_MEASURER_ASSESSMENT = 'Bird Measurer (Assessment)'
LGB_INTEGER_PARAMS = ['max_depth', 'max_bin', 'num_leaves', 'min_child_samples', 'n_splits', 'subsample_freq']


def fillna0(df):
    return df.fillna(0.0)


def preprocess_events():
    with open('event_props.json', encoding='utf-8') as f:
        event_ids_to_props = json.load(f)

    def _postprocessing(game_sessions):
        for idx in range(len(game_sessions)):
            event_codes = game_sessions[idx]['event_codes']
            event_ids = game_sessions[idx]['event_ids']
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

            correct, uncorrect = game_sessions[idx]['correct_attempts'], game_sessions[idx]['uncorrect_attempts']
            if correct == 0 and uncorrect == 0:
                game_sessions[idx]['accuracy_rate'] = 0.0
            else:
                game_sessions[idx]['accuracy_rate'] = correct / float(correct + uncorrect)

            if correct == 0 and uncorrect == 0:
                pass
            elif correct == 1 and uncorrect == 0:
                game_sessions[idx]['accuracy_group_3'] = 1
            elif correct == 1 and uncorrect == 1:
                game_sessions[idx]['accuracy_group_2'] = 1
            elif correct == 1 and uncorrect >= 2:
                game_sessions[idx]['accuracy_group_1'] = 1
            else:
                game_sessions[idx]['accuracy_group_0'] = 1

            game_sessions[idx] = {
                **game_sessions[idx],
                **dict(event_codes),
                **dict(event_ids),
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
                        # 'event_data_props': defaultdict(int)
                    }

                game_session = game_sessions[row['game_session']]

                game_session['event_count'] = max(game_session['event_count'], int(row['event_count']))
                game_session['game_times'].append(int(row['game_time']))

                event_data = json.loads(row['event_data'])
                event_code = int(row['event_code'])

                game_session['event_codes'][f'cnt_event_code_{event_code}'] += 1

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
                if event_id in important_event_ids:
                    game_session['event_ids'][f'cnt_event_id_{event_id}'] += 1

                if idx % 10000 == 0:
                    print(idx)

        df_data = fillna0(pd.DataFrame(_postprocessing(list(game_sessions.values()))))
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
    def _add_time_features(df):
        timestamp = pd.to_datetime(df['timestamp'])
        df['timestamp'] = (timestamp.astype(int) / 10 ** 9).astype(int)
        df['assessment_hour'] = timestamp.dt.hour
        df['assessment_dayofweek'] = timestamp.dt.dayofweek

    def _get_aggregated_column_name(column):
        if not isinstance(column, tuple):
            return column

        if column[1] == 'first' or column[1] == '':
            return column[0]

        return '_'.join(column)

    def _compute_features(df_data, prefix, df_assessments):
        game_sessions = []

        df_assessments = df_assessments.sort_values(by='installation_id')
        installation_id_to_game_sessions = dict(tuple(df_data.groupby('installation_id')))

        total_assessments = df_assessments.shape[0]
        for idx, assessment in df_assessments.iterrows():
            installation_id = assessment['installation_id']
            start_timestamp = assessment['timestamp']

            game_sessions_for_installation_id = installation_id_to_game_sessions[installation_id]

            previous_game_sessions: pd.DataFrame = game_sessions_for_installation_id[
                game_sessions_for_installation_id['timestamp'] < start_timestamp
                ].copy(deep=True)

            if previous_game_sessions.shape[0] == 0:
                game_sessions.append(
                    pd.DataFrame({
                        'title': '__no_title',
                        'type': '__no_type',
                        'world': '__no_world',
                        'installation_id': installation_id,
                        'assessment_game_session': assessment['game_session'],
                        'assessment_title': assessment['title'],
                        'assessment_world': assessment['world'],
                        'assessment_most_common_title_accuracy_group': assessment[
                            'assessment_most_common_title_accuracy_group'],
                        'assessment_dayofweek': assessment['assessment_dayofweek'],
                        'assessment_hour': assessment['assessment_hour'],
                    }, index=[0])
                )
                continue

            # Previous attempts for current assessment
            # Which attempt, accuracy groups for current assessment, correct/incorrect/rate attempts
            previous_attempts_agg = previous_game_sessions[
                previous_game_sessions['title'] == assessment['title']
                ].agg(aggregate_assessment_game_session_columns).reset_index()

            for _, row in previous_attempts_agg.iterrows():
                for column in aggregate_assessment_game_session_columns.keys():
                    previous_game_sessions[f'assessment_previous_{column}_{row["index"]}'] = row[column]

            # Everything with the assessment_* prefix is related to
            # the assessment we are predicting the accuracy_group for.
            # Everything else is aggregated from game sessions that happened before the assessment.
            previous_game_sessions['installation_id'] = installation_id

            copy_from_assessment = {
                'game_session': 'assessment_game_session',
                'assessment_most_common_title_accuracy_group': 'assessment_most_common_title_accuracy_group',
                'assessment_dayofweek': 'assessment_dayofweek',
                'assessment_hour': 'assessment_hour',
                'title': 'assessment_title',
                'world': 'assessment_world'
            }
            for k, v in copy_from_assessment.items():
                previous_game_sessions[v] = assessment[k]

            game_sessions.append(previous_game_sessions)

            # if idx == 100:
            #     break

            if idx % 100 == 0:
                print(f'Row {idx + 1}/{total_assessments} done')

        df_final = pd.concat(game_sessions, ignore_index=True, sort=False)
        assessment_game_sessions = df_final['type'] == 'Assessment'
        df_final = pd.get_dummies(df_final, columns=['title', 'type', 'world'], prefix='ttw')

        _agg_cols = {
            **aggregate_game_sessions_columns,

            **{column: 'first'
               for column in df_final.columns
               if column.startswith('assessment') and column != 'assessment_game_session'},

            **{column: ['std', 'mean', 'median', 'max']
               for column in df_final.columns if column.startswith('cnt')},

            # Sum dummy columns (ttw_(title*), etc.)
            **{column: 'sum'
               for column in df_final.columns if column.startswith('ttw')}
        }

        df_final_aggregated = df_final[['installation_id', 'assessment_game_session', *_agg_cols.keys()]] \
            .groupby(['installation_id', 'assessment_game_session']) \
            .agg(_agg_cols) \
            .reset_index()

        df_final_assessments_aggregated = df_final[assessment_game_sessions] \
            [['installation_id', 'assessment_game_session', *aggregate_assessment_game_session_columns.keys()]] \
            .groupby(['installation_id', 'assessment_game_session']) \
            .agg(aggregate_assessment_game_session_columns) \
            .reset_index()

        df_final_aggregated.columns = [
            _get_aggregated_column_name(column) for column in df_final_aggregated.columns
        ]

        df_final_assessments_aggregated.columns = [
            _get_aggregated_column_name(column) for column in df_final_assessments_aggregated.columns
        ]

        df_final_aggregated = df_final_aggregated.merge(
            df_final_assessments_aggregated,
            on=['installation_id', 'assessment_game_session'],
            how='left'
        )

        print('Writing features...')
        df_final_aggregated.to_csv(f'preprocessed-data/{prefix}_features.csv', index=False)

    aggregate_game_sessions_columns = {
        'game_time': ['std', 'mean', 'median', 'max'],
        'event_count': ['std', 'mean', 'median', 'max'],
        'game_time_mean_diff': ['std', 'mean', 'median', 'max'],
        'game_time_std_diff': ['std', 'mean', 'median', 'max'],
    }

    aggregate_assessment_game_session_columns = {
        'correct_attempts': ['std', 'mean', 'median', 'max'],
        'uncorrect_attempts': ['std', 'mean', 'median', 'max'],
        'accuracy_rate': ['std', 'mean', 'median', 'max'],
    }

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
    # Make sure test columns are in the correct order
    df_test_features = df_test_features.drop(columns=columns_to_drop)[df_train_features.columns]

    return (
        df_train_features,
        df_test_features,
        y_correct,
        y_uncorrect,
        y_accuracy_rate,
        y_accuracy_group,
        train_installation_ids,
        test_installation_ids
    )


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


def integer_encode_params(params):
    for param, value in params.items():
        if param in LGB_INTEGER_PARAMS:
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
    (
        df_train_features,
        df_test_features,
        y_correct,
        _,
        _,
        y_accuracy_group,
        train_installation_ids,
        _
    ) = get_train_test_features()

    df_train, df_test = label_encode_categorical_features(df_train_features, df_test_features)
    model_params = integer_encode_params(params)

    meta_train, meta_test = cv_with_oof_predictions(
        get_lgbm_classifier('binary', 'auc'),
        roc_auc_score,
        df_train,
        df_test,
        y_correct,
        y_accuracy_group,
        train_installation_ids,
        model_params
    )

    return meta_train[:, 1].reshape(-1, 1), meta_test[:, 1].reshape(-1, 1)


def fit_predict_accuracy_group_regression_model(params):
    (
        df_train_features,
        df_test_features,
        _,
        _,
        _,
        y_accuracy_group,
        train_installation_ids,
        _
    ) = get_train_test_features()

    df_train, df_test = label_encode_categorical_features(df_train_features, df_test_features)
    model_params = integer_encode_params(params)

    return cv_with_oof_predictions(
        get_lgbm_regressor('regression', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        df_train,
        df_test,
        y_accuracy_group,
        y_accuracy_group,
        train_installation_ids,
        model_params,
        predict_proba=False,
        n_predicted_features=1
    )


def fit_predict_accuracy_rate_regression_model(params):
    (
        df_train_features,
        df_test_features,
        _,
        _,
        y_accuracy_rate,
        y_accuracy_group,
        train_installation_ids,
        _
    ) = get_train_test_features()

    df_train, df_test = label_encode_categorical_features(df_train_features, df_test_features)
    model_params = integer_encode_params(params)

    return cv_with_oof_predictions(
        get_lgbm_regressor('regression', 'rmse'),
        lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)),
        df_train,
        df_test,
        y_accuracy_rate,
        y_accuracy_group,
        train_installation_ids,
        model_params,
        predict_proba=False,
        n_predicted_features=1
    )


def fit_predict_ordinal_model(params):
    train_predictions = []
    test_predictions = []
    for target in [0, 1, 2]:
        (
            df_train_features,
            df_test_features,
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

        df_train, df_test = label_encode_categorical_features(df_train_features, df_test_features)
        model_params = integer_encode_params(params[target])

        train_prediction, test_prediction = cv_with_oof_predictions(
            get_lgbm_classifier('binary', 'auc'),
            roc_auc_score,
            df_train,
            df_test,
            y_accuracy_group,
            y_accuracy_group,
            train_installation_ids,
            model_params,
        )

        train_predictions.append(train_prediction[:, 1])
        test_predictions.append(test_prediction[:, 1])

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


def fit_predict_accuracy_group_classification_model(params):
    (
        df_train_features,
        df_test_features,
        _,
        _,
        _,
        y_accuracy_group,
        train_installation_ids,
        _
    ) = get_train_test_features()

    df_train, df_test = label_encode_categorical_features(df_train_features, df_test_features)
    model_params = integer_encode_params(params)

    return cv_with_oof_predictions(
        get_lgbm_classifier('multiclass', 'multi_logloss'),
        lambda y_true, y_pred: cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        df_train,
        df_test,
        y_accuracy_group,
        y_accuracy_group,
        train_installation_ids,
        model_params,
        predict_proba=False,
        n_predicted_features=1
    )


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
            'thresholds': [0.4, 0.8, 0.95],
            'params': {'colsample_bytree': 0.47984635556648836,
                       'learning_rate': 0.6897993073933458,
                       'max_bin': 27.742730692896608,
                       'max_depth': 6.028138981293016,
                       'min_child_samples': 506.6545097913775,
                       'min_child_weight': 11.760784619154952,
                       'num_leaves': 5.955211245841726,
                       'reg_alpha': 21.967165615335805,
                       'reg_lambda': 23.713472502100664,
                       'subsample': 0.7810425506567095}
        },
        {
            'name': 'accuracy_group_regression',
            'thresholds': [0.5, 1.5, 2.5],
            'params': {'colsample_bytree': 0.5228040860926025,
                       'learning_rate': 0.07724056467062121,
                       'max_bin': 498.11463796877655,
                       'max_depth': 14.40392412216746,
                       'min_child_samples': 61.82114461202193,
                       'min_child_weight': 383.67007431244184,
                       'num_leaves': 16.270095354880308,
                       'reg_alpha': 5.4925035589459155,
                       'reg_lambda': 9.14097666376884,
                       'subsample': 0.9478851516255011}
        },
        {
            'name': 'accuracy_rate_regression',
            'thresholds': [0.25, 0.5, 0.75],
            'params': {'colsample_bytree': 0.5509332350577756,
                       'learning_rate': 0.024398654076713232,
                       'max_bin': 478.1751122178361,
                       'max_depth': 4.848488969027337,
                       'min_child_samples': 59.09089537333169,
                       'min_child_weight': 57.268881739939026,
                       'num_leaves': 7.054775941496443,
                       'reg_alpha': 1.0094867667887906,
                       'reg_lambda': 19.6864276401854,
                       'subsample': 0.8807692818134909}
        },
        {
            'name': 'accuracy_group_ordinal',
            'params': [
                {'colsample_bytree': 0.42799512532781714,
                 'learning_rate': 0.9546705201666181,
                 'max_bin': 7.972568171233059,
                 'max_depth': 15.848704853297166,
                 'min_child_samples': 1262.4562530652881,
                 'min_child_weight': 215.71669482798504,
                 'num_leaves': 498.9289995496746,
                 'reg_alpha': 4.3268202556286415,
                 'reg_lambda': 3.297550625252506,
                 'subsample': 0.8474822340940421},
                {'colsample_bytree': 0.4775890403241546,
                 'learning_rate': 0.33396992615181276,
                 'max_bin': 27.082980272944475,
                 'max_depth': 15.378196609933338,
                 'min_child_samples': 813.0748044866655,
                 'min_child_weight': 296.38469121726035,
                 'num_leaves': 195.6269480155727,
                 'reg_alpha': 1.0461139915174535,
                 'reg_lambda': 28.317465698124646,
                 'subsample': 0.8182873912569566},
                {'colsample_bytree': 0.1268668500934963,
                 'learning_rate': 0.022840475127116164,
                 'max_bin': 4.83437492826646,
                 'max_depth': 15.663093790434736,
                 'min_child_samples': 498.255455267949,
                 'min_child_weight': 274.15506079865867,
                 'num_leaves': 414.4567625152621,
                 'reg_alpha': 13.85533926180778,
                 'reg_lambda': 24.158347480344702,
                 'subsample': 0.7484079419356578}
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
            elif model_desc['name'] == 'accuracy_group_classification':
                base_features.append(fit_predict_accuracy_group_classification_model(model_desc['params']))

        meta_train_features = np.concatenate([_train for (_train, _) in base_features], axis=1)
        meta_test_features = np.concatenate([_test for (_, _test) in base_features], axis=1)

        (_, _, _, _, _, y_accuracy_group, train_installation_ids, test_installation_ids) = get_train_test_features()

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

        meta_train_thresholded = {}
        meta_test_thresholded = {}
        for model_desc in models:
            if 'thresholds' in model_desc:
                or_ = OptimizedRounder(model_desc['thresholds'], [0, 1, 2, 3])
                or_.fit(df_meta_train_features[model_desc['name']], target)

                train_prediction = or_.predict(df_meta_train_features[model_desc['name']], or_.coefficients())
                test_prediction = or_.predict(df_meta_test_features[model_desc['name']], or_.coefficients())

                print(
                    model_desc['name'],
                    or_.coefficients(),
                    cohen_kappa_score(train_prediction, target, weights='quadratic')
                )

                meta_train_thresholded[model_desc['name']] = train_prediction.values
                meta_test_thresholded[model_desc['name']] = test_prediction.values
            else:
                meta_train_thresholded[model_desc['name']] = df_meta_train_features[
                    model_desc['name']].values.astype(int)
                meta_test_thresholded[model_desc['name']] = df_meta_test_features[model_desc['name']].values.astype(int)

                print(
                    model_desc['name'],
                    cohen_kappa_score(meta_train_thresholded[model_desc['name']], target, weights='quadratic')
                )

        df_train_prediction = pd.DataFrame(meta_train_thresholded)
        df_test_prediction = pd.DataFrame(meta_test_thresholded)

        train_prediction = []
        for _, row in df_train_prediction.iterrows():
            pred = np.bincount(row.values.astype(int), minlength=4).argmax()
            train_prediction.append(pred)

        test_prediction = []
        for _, row in df_test_prediction.iterrows():
            pred = np.bincount(row.values.astype(int), minlength=4).argmax()
            test_prediction.append(pred)

        print(cohen_kappa_score(train_prediction, target, weights='quadratic'))

        pd.DataFrame.from_dict({
            'installation_id': test_installation_ids,
            'accuracy_group': test_prediction
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

output_submission()
