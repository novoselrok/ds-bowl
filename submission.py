import csv
import json
import os
import gc
from collections import defaultdict

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

BIRD_MEASURER_ASSESSMENT = 'Bird Measurer (Assessment)'


def fillna0(df):
    return df.fillna(0.0)


def preprocess_events():
    with open(EVENT_PROPS_JSON, encoding='utf-8') as f:
        event_ids_to_props = json.load(f)

    def _postprocessing(game_sessions):
        for idx in range(len(game_sessions)):
            event_codes = game_sessions[idx]['event_codes']
            event_data_props = game_sessions[idx]['event_data_props']

            del game_sessions[idx]['event_codes']
            del game_sessions[idx]['event_data_props']

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
                **dict(event_data_props)
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
                        'game_time': 0,
                        'correct_attempts': 0,
                        'uncorrect_attempts': 0,
                        'title': row['title'],
                        'type': row['type'],
                        'world': row['world'],
                        'event_codes': defaultdict(int),
                        'event_data_props': defaultdict(int)
                    }

                game_session = game_sessions[row['game_session']]

                game_session['event_count'] = max(game_session['event_count'], int(row['event_count']))
                game_session['game_time'] = max(game_session['game_time'], int(row['game_time']))

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
                if event_id in event_ids_to_props:
                    for prop in event_ids_to_props[event_id]:
                        game_session['event_data_props'][f'event_data_prop_{event_id}_{prop}'] += event_data[prop]

                if idx % 10000 == 0:
                    print(idx)

        df_data = pd.DataFrame(_postprocessing(list(game_sessions.values())))
        df_data.fillna(0.0, inplace=True)
        df_data.to_csv(f'preprocessed-data/{prefix}_game_sessions.csv', index=False)

    os.makedirs('preprocessed-data', exist_ok=True)

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
                continue

            # Previous attempts for current assessment
            # Which attempt, accuracy groups for current assessment, correct/incorrect/rate attempts
            previous_attempts_agg = fillna0(
                previous_game_sessions[
                    previous_game_sessions['title'] == assessment['title']
                ].agg(aggregate_game_sessions_columns).reset_index()
            )

            for _, row in previous_attempts_agg.iterrows():
                for column in aggregate_game_sessions_columns.keys():
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

        df_final = fillna0(pd.concat(game_sessions, ignore_index=True, sort=False))
        df_final = pd.get_dummies(df_final, columns=['title', 'type', 'world'], prefix='cnt')

        aggregate_columns = {
            **aggregate_game_sessions_columns,

            **{column: 'first'
               for column in df_final.columns if
               column.startswith('assessment') and column != 'assessment_game_session'},

            **{column: ['sum', 'mean', 'max', 'std', 'median']
               for column in df_final.columns if column.startswith('event_data_prop')},

            # Sum dummy columns (cnt_train_*, cnt_type_*, cnt_world_*, cnt_event_code_*)
            **{column: ['sum', 'mean']
               for column in df_final.columns if column.startswith('cnt')}
        }

        df_final_aggregated = fillna0(
            df_final[
                ['installation_id', 'assessment_game_session', *aggregate_columns.keys()]
            ].groupby(['installation_id', 'assessment_game_session']).agg(aggregate_columns).reset_index()
        )
        df_final_aggregated.columns = [_get_aggregated_column_name(column) for column in df_final_aggregated.columns]

        print('Writing features...')
        df_final_aggregated.to_csv(f'preprocessed-data/{prefix}_features.csv', index=False)

    aggregate_game_sessions_columns = {
        'game_time': ['sum', 'mean', 'max', 'std', 'median'],
        'event_count': ['sum', 'mean', 'max', 'std', 'median'],
        'correct_attempts': ['sum', 'mean', 'max', 'std', 'median'],
        'uncorrect_attempts': ['sum', 'mean', 'max', 'std', 'median'],
        'accuracy_rate': ['sum', 'mean', 'max', 'std', 'median'],
        'accuracy_group_0': ['sum', 'mean'],
        'accuracy_group_1': ['sum', 'mean'],
        'accuracy_group_2': ['sum', 'mean'],
        'accuracy_group_3': ['sum', 'mean'],
    }

    df_train_labels = pd.read_csv(TRAIN_LABELS_CSV)
    df_train = pd.read_csv('preprocessed-data/train_game_sessions.csv')

    title_to_world = df_train[['title', 'world']].groupby('title').agg('first').apply(list).to_dict()
    # Add assessment world
    df_train_labels['world'] = df_train_labels.apply(
        lambda assessment: title_to_world['world'][assessment['title']],
        axis=1
    )

    print('Preparing train data...')
    # Add most common title accuracy group to assessment
    title_to_acc_group_dict = dict(
        df_train_labels.groupby('title')['accuracy_group'].agg(lambda x: x.value_counts().index[0])
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


def get_train_test_features():
    columns_to_drop = [
        'installation_id',
        'assessment_game_session',
    ]

    df_train_features = pd.read_csv('preprocessed-data/train_features.csv')
    df_test_features = pd.read_csv('preprocessed-data/test_features.csv')

    df_train_labels = pd.read_csv(
        TRAIN_LABELS_CSV,
        usecols=['installation_id', 'game_session', 'accuracy_group']
    ).rename({'game_session': 'assessment_game_session', 'accuracy_group': 'target'}, axis=1)

    # Add target to train features
    df_train_features = df_train_features.merge(df_train_labels, on=['installation_id', 'assessment_game_session'])

    # Extract target and installation ids
    y_target = df_train_features['target'].values
    train_installation_ids = df_train_features['installation_id']
    test_installation_ids = df_test_features['installation_id']

    # Encode categorical features
    for column in categorical_features:
        lb = LabelEncoder()
        train_column = df_train_features[column]
        test_column = df_test_features[column]
        lb.fit(train_column)
        df_train_features[column] = lb.transform(train_column)
        df_test_features[column] = lb.transform(test_column)

    df_train_features = df_train_features.drop(columns=columns_to_drop + ['target'])
    # Make sure test columns are in the correct order
    df_test_features = df_test_features.drop(columns=columns_to_drop)[df_train_features.columns]

    return (
        df_train_features,
        df_test_features,
        y_target,
        train_installation_ids,
        test_installation_ids
    )


def output_submission():
    def _train():
        n_splits = 8
        kf = GroupKFold(n_splits=n_splits)

        scores = []
        y_test = np.zeros((X_test.shape[0], 4))
        for fold, (train_split, test_split) in enumerate(kf.split(X, y_target, train_installation_ids)):
            print(f'Starting fold {fold}...')

            x_train, x_val, y_train, y_val = X[train_split], X[test_split], y_target[train_split], y_target[
                test_split]

            params = {
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
            model = LGBMClassifier(
                random_state=2019,
                n_estimators=100000,
                num_classes=4,
                objective='multiclass',
                metric='multi_logloss',
                n_jobs=-1,
                **params,
            )

            model.fit(
                x_train, y_train,
                early_stopping_rounds=500,
                eval_set=[(x_train, y_train), (x_val, y_val)],
                verbose=100,
                feature_name=feature_names,
                categorical_feature=categorical_features
            )

            y_test += model.predict_proba(X_test) / n_splits

            # Diagnostic
            pred = model.predict_proba(x_val).argmax(axis=1)
            score = cohen_kappa_score(y_val, pred, weights='quadratic')
            scores.append(score)
            print(score)

            print('Fold done.')

        print(np.mean(scores))
        pd.DataFrame.from_dict({
            'installation_id': test_installation_ids,
            'accuracy_group': y_test.argmax(axis=1)
        }).to_csv('submission.csv', index=False)

    (
        df_train_features,
        df_test_features,
        y_target,
        train_installation_ids,
        test_installation_ids
    ) = get_train_test_features()

    feature_names = df_train_features.columns.tolist()
    # Features matrix
    X = df_train_features.values
    X_test = df_test_features.values

    _train()


categorical_features = [
    'assessment_title',
    'assessment_world',
]

TRAIN_CSV = 'data/train.csv'
TRAIN_LABELS_CSV = 'data/train_labels.csv'
TEST_CSV = 'data/test.csv'
EVENT_PROPS_JSON = 'event_props.json'
CORRELATED_FEATURES_JSON = 'correlated_features.json'
FEATURE_IMPORTANCES_CSV = 'feature_importances.csv'
