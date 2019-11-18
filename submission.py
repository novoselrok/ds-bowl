import csv
import json
import os
import gc
import random
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import cohen_kappa_score
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


def fillna0(df):
    return df.fillna(0.0)


def preprocess_events():
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


def label_encode_categorical_features(df_train, df_test):
    for column in categorical_features:
        lb = LabelEncoder()
        train_column = df_train[column]
        test_column = df_test[column]
        lb.fit(train_column)
        df_train[column] = lb.transform(train_column)
        df_test[column] = lb.transform(test_column)

        df_train[column] = df_train[column].astype('category')
        df_test[column] = df_test[column].astype('category')

    return df_train, df_test


def one_hot_encode_categorical_features(df_train, df_test):
    df_train = pd.get_dummies(df_train, columns=categorical_features)
    df_test = pd.get_dummies(df_test, columns=categorical_features)[df_train.columns]

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
        usecols=['installation_id', 'game_session', 'num_correct', 'num_incorrect', 'accuracy_group']
    ).rename({'game_session': 'assessment_game_session'}, axis=1)

    # Add target to train features
    df_train_features = df_train_features.merge(df_train_labels, on=['installation_id', 'assessment_game_session'])

    # Extract target and installation ids
    y_correct = df_train_features['num_correct'].values
    y_uncorrect = df_train_features['num_incorrect'].values
    y_accuracy_group = df_train_features['accuracy_group'].values
    train_installation_ids = df_train_features['installation_id']
    test_installation_ids = df_test_features['installation_id']

    df_train_features = df_train_features.drop(
        columns=columns_to_drop + ['num_correct', 'num_incorrect', 'accuracy_group']
    )
    # Make sure test columns are in the correct order
    df_test_features = df_test_features.drop(columns=columns_to_drop)[df_train_features.columns]

    return (
        df_train_features,
        df_test_features,
        y_correct,
        y_uncorrect,
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


def get_correct_attempts_clf(model_params):
    return LGBMClassifier(
        random_state=2019,
        n_estimators=5000,
        n_jobs=-1,
        objective='binary',
        metric='binary_logloss',
        **model_params
    )


def get_uncorrect_attempts_reg(model_params):
    return LGBMRegressor(
        random_state=2019,
        n_estimators=5000,
        n_jobs=-1,
        **model_params
    )


def fit_model(model, x_train, y_train, x_val, y_val):
    model.fit(
        x_train, y_train,
        early_stopping_rounds=100,
        eval_set=[(x_val, y_val)],
        verbose=0,
    )


def attempts_to_group(correct_proba, uncorrect):
    uncorrect = uncorrect if uncorrect >= 0 else 0
    if correct_proba[0] > 0.4:
        return 0

    if correct_proba[0] < 0.07:
        return 3

    if uncorrect < 1:
        return 3
    if uncorrect < 2:
        return 2

    return 1


def output_submission():
    def _train():
        n_splits = 5

        df_train, df_test = label_encode_categorical_features(df_train_features, df_test_features)

        scores = []

        test_preds = np.zeros((df_test.shape[0], n_splits))

        kf = stratified_group_k_fold(None, y_accuracy_group, train_installation_ids, n_splits, seed=2019)
        for fold, (train_split, test_split) in enumerate(kf):
            print(f'Starting fold {fold}...')

            x_train, x_val = df_train.iloc[train_split, :], df_train.iloc[test_split, :]
            y_correct_train, y_correct_val = y_correct[train_split], y_correct[test_split]
            y_uncorrect_train, y_uncorrect_val = y_uncorrect[train_split], y_uncorrect[test_split]

            correct_attempts_clf = get_correct_attempts_clf(clf_correct_attempts_params)
            uncorrect_attempts_reg = get_uncorrect_attempts_reg(reg_uncorrect_attempts_params)

            fit_model(correct_attempts_clf, x_train, y_correct_train, x_val, y_correct_val)
            fit_model(uncorrect_attempts_reg, x_train, y_uncorrect_train, x_val, y_uncorrect_val)

            correct_attempts_pred = correct_attempts_clf.predict_proba(df_test)
            uncorrect_attempts_pred = uncorrect_attempts_reg.predict(df_test)
            accuracy_group_pred = [
                attempts_to_group(c, u) for c, u in zip(correct_attempts_pred, uncorrect_attempts_pred)
            ]
            test_preds[:, fold] = accuracy_group_pred

            correct_attempts_val_pred = correct_attempts_clf.predict_proba(x_val)
            uncorrect_attempts_val_pred = uncorrect_attempts_reg.predict(x_val)
            accuracy_group_val_pred = [
                attempts_to_group(c, u) for c, u in zip(correct_attempts_val_pred, uncorrect_attempts_val_pred)
            ]

            score = cohen_kappa_score(y_accuracy_group[test_split], accuracy_group_val_pred, weights='quadratic')
            scores.append(score)

        print('Final score:', np.mean(scores))

        accuracy_groups = []
        for row in test_preds:
            cnts = np.bincount(row.astype(int), minlength=4)
            group = cnts.argmax()
            accuracy_groups.append(group)

        pd.DataFrame.from_dict({
            'installation_id': test_installation_ids,
            'accuracy_group': accuracy_groups
        }).to_csv('submission.csv', index=False)

    (
        df_train_features,
        df_test_features,
        y_correct,
        y_uncorrect,
        y_accuracy_group,
        train_installation_ids,
        test_installation_ids
    ) = get_train_test_features()

    clf_correct_attempts_params = {
        "colsample_bytree": 0.48800692625786024,
        "learning_rate": 0.1020388475140511,
        "max_bin": 253,
        "max_depth": 6,
        "min_child_samples": 55,
        "min_child_weight": 96.41746550933172,
        "num_leaves": 227,
        "reg_alpha": 2.5075658866930826,
        "reg_lambda": 26.531548533942157,
        "subsample": 0.8135279342738913,
        "subsample_freq": 1,
    }

    reg_uncorrect_attempts_params = {
        "colsample_bytree": 0.22442543382762706,
        "learning_rate": 0.1449358327239709,
        "max_bin": 11,
        "max_depth": 5,
        "min_child_samples": 804,
        "min_child_weight": 986.686500096391,
        "num_leaves": 38,
        "reg_alpha": 25.08821535738595,
        "reg_lambda": 29.658691679324352,
        "subsample": 0.6076212307221736,
        "subsample_freq": 4
    }

    _train()


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
