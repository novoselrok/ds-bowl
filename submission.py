import json
import os
import gc

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder


def preprocess_events():
    def _extract_props_from_event_data(df_events):
        event_data_props_columns = []
        for event_id, props in event_ids_to_props.items():
            if len(props) == 0:
                continue

            for prop in props:
                event_data_props_columns.append(f'event_data_prop_{event_id}_{prop}')

        df_events_per_game_session = df_events.groupby('game_session')
        game_sessions_len = len(df_events_per_game_session)

        columns = {column: np.zeros((game_sessions_len,)) for column in event_data_props_columns}
        columns['game_session'] = np.empty((game_sessions_len,), dtype=object)

        group_idx = 0
        for idx, event_group in df_events_per_game_session:
            aggregated_game_session = {column: 0.0 for column in event_data_props_columns}
            aggregated_game_session['game_session'] = event_group.iloc[0]['game_session']

            for _, event in event_group.iterrows():
                event_id = event['event_id']
                event_data = json.loads(event['event_data'])

                if event_id not in event_ids_to_props:
                    continue

                props = event_ids_to_props[event_id]
                for prop in props:
                    aggregated_game_session[f'event_data_prop_{event_id}_{prop}'] += event_data[prop]

            for key, value in aggregated_game_session.items():
                columns[key][group_idx] = value

            if group_idx % 1000 == 0:
                print(f'Done group {group_idx}/{game_sessions_len}')

            group_idx += 1

        return pd.DataFrame(columns)

    def _preprocess(df_data, prefix):
        is_assessment = df_data['type'] == 'Assessment'

        is_bird_measurer_attempt = (df_data['title'] == 'Bird Measurer (Assessment)') & (df_data['event_code'] == 4110)
        is_non_bird_measurer_attempt = (df_data['title'] != 'Bird Measurer (Assessment)') & (
                df_data['event_code'] == 4100)
        is_assessment_attempt = is_assessment & (is_bird_measurer_attempt | is_non_bird_measurer_attempt)

        is_correct_attempt = df_data['event_data'].str.contains('"correct":true')
        is_uncorrect_attempt = df_data['event_data'].str.contains('"correct":false')

        df_data['is_correct_attempt'] = (is_assessment_attempt & is_correct_attempt).astype(int)
        df_data['is_uncorrect_attempt'] = (is_assessment_attempt & is_uncorrect_attempt).astype(int)

        _extract_props_from_event_data(df_data[['game_session', 'event_id', 'event_data']]) \
            .to_csv(f'preprocessed-data/{prefix}_event_data_props_per_game_session.csv', index=False)

        df_data = df_data.drop(columns=['event_data'])
        df_data.to_csv(f'preprocessed-data/{prefix}_preprocessed.csv', index=False)

    with open(EVENT_PROPS_JSON, encoding='utf-8') as f:
        event_ids_to_props = json.load(f)

    os.makedirs('preprocessed-data', exist_ok=True)

    df_train = pd.read_csv(TRAIN_CSV)
    df_train_labels = pd.read_csv(TRAIN_LABELS_CSV)

    # Only keep installation ids from train labels
    installation_ids = df_train_labels['installation_id'].unique()
    train_event_codes = set(df_train['event_code'].unique())
    df_train = df_train[df_train['installation_id'].isin(installation_ids)].reset_index(drop=True)

    print('Preprocessing train...')
    _preprocess(df_train, 'train')

    del df_train
    gc.collect()

    print('Preprocessing test...')
    df_test = pd.read_csv(TEST_CSV)
    _preprocess(df_test, 'test')

    print('Saving event codes...')
    event_codes = train_event_codes & set(df_test['event_code'].unique())
    pd.DataFrame({'event_code': list(event_codes)}).to_csv('preprocessed-data/event_codes.csv', index=False)


def feature_engineering():
    def _add_time_features(df):
        timestamp = pd.to_datetime(df['timestamp'])
        df['timestamp'] = (timestamp.astype(int) / 10 ** 9).astype(int)
        df['assessment_hour'] = timestamp.dt.hour
        df['assessment_dayofweek'] = timestamp.dt.dayofweek

    def _get_default_accuracy_groups_dict():
        return {0: 0, 1: 0, 2: 0, 3: 0}

    def _get_aggregated_column_name(column):
        if not isinstance(column, tuple):
            return column

        if column[1] == 'first' or column[1] == '':
            return column[0]

        return '_'.join(column)

    def _compute_accuracy_rate(row):
        correct, uncorrect = row['is_correct_attempt'], row['is_uncorrect_attempt']
        if correct == 0 and uncorrect == 0:
            return 0.0

        return correct / float(correct + uncorrect)

    def _compute_accuracy_group(row):
        correct, uncorrect = row['is_correct_attempt'], row['is_uncorrect_attempt']
        if correct == 0 and uncorrect == 0:
            return np.nan

        if correct == 1 and uncorrect == 0:
            return 3
        elif correct == 1 and uncorrect == 1:
            return 2
        elif correct == 1 and uncorrect >= 2:
            return 1

        # Never correctly solved
        return 0

    def _compute_features(df_data, prefix, df_assessments, df_event_data_props_per_game_session):
        game_sessions = []

        df_assessments = df_assessments.sort_values(by='installation_id')
        total_assessments = df_assessments.shape[0]

        installation_id_to_events = dict(tuple(df_data.groupby('installation_id')))

        for idx, assessment in df_assessments.iterrows():
            installation_id = assessment['installation_id']
            start_timestamp = assessment['timestamp']

            events_for_installation_id = installation_id_to_events[installation_id]

            previous_events: pd.DataFrame = events_for_installation_id[
                events_for_installation_id['timestamp'] < start_timestamp]

            present_event_codes = ['cnt_event_code_' + str(column)
                                   for column in list(previous_events['event_code'].unique())]
            previous_events = pd.get_dummies(previous_events, columns=['event_code'], prefix='cnt_event_code')

            if previous_events.shape[0] == 0:
                game_sessions.append(
                    pd.DataFrame({
                        'game_time': 0,
                        'event_count': 0,
                        'correct_attempts': 0,
                        'uncorrect_attempts': 0,
                        'accuracy_rate': 0.0,
                        'accuracy_group': np.nan,
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
                        **{column: 0 for column in event_codes}
                    }, index=[0])
                )
                continue

            aggregated_game_sessions = previous_events[
                ['game_session', 'game_time', 'event_count', 'is_correct_attempt', 'is_uncorrect_attempt', 'title',
                 'type', 'world', *present_event_codes]
            ].groupby('game_session').agg(
                {'game_time': 'max', 'event_count': 'max', 'is_correct_attempt': 'sum', 'is_uncorrect_attempt': 'sum',
                 'title': 'first', 'type': 'first', 'world': 'first',
                 **{column: 'sum' for column in present_event_codes}}
            ).reset_index()

            aggregated_game_sessions['accuracy_group'] = aggregated_game_sessions \
                .apply(_compute_accuracy_group, axis=1)

            aggregated_game_sessions['accuracy_rate'] = aggregated_game_sessions \
                .apply(_compute_accuracy_rate, axis=1)

            aggregated_game_sessions.rename({
                'is_correct_attempt': 'correct_attempts',
                'is_uncorrect_attempt': 'uncorrect_attempts',
            }, axis=1, inplace=True)

            # Previous attempts for current assessment
            # Which attempt, accuracy groups for current assessment, correct/incorrect/rate attempts
            previous_attempts_agg_columns = {
                'game_time': ['sum', 'mean', 'max', 'std', 'median'],
                'event_count': ['sum', 'mean', 'max', 'std', 'median'],
                'correct_attempts': ['sum', 'mean', 'max', 'std', 'median'],
                'uncorrect_attempts': ['sum', 'mean', 'max', 'std', 'median'],
                'accuracy_rate': ['sum', 'mean', 'max', 'std', 'median'],
            }
            previous_attempts_agg = aggregated_game_sessions[
                aggregated_game_sessions['title'] == assessment['title']
                ].agg(previous_attempts_agg_columns).reset_index()
            previous_attempts_agg.fillna(0.0, inplace=True)

            for _, row in previous_attempts_agg.iterrows():
                for column in previous_attempts_agg_columns.keys():
                    aggregated_game_sessions[f'assessment_previous_{column}_{row["index"]}'] = row[column]

            # Everything with the assessment_* prefix is related to
            # the assessment we are predicting the accuracy_group for.
            # Everything else is aggregated from game sessions that happened before the assessment.
            aggregated_game_sessions['installation_id'] = installation_id
            aggregated_game_sessions['assessment_most_common_title_accuracy_group'] = assessment[
                'assessment_most_common_title_accuracy_group']
            aggregated_game_sessions['assessment_dayofweek'] = assessment['assessment_dayofweek']
            aggregated_game_sessions['assessment_hour'] = assessment['assessment_hour']
            aggregated_game_sessions['assessment_game_session'] = assessment['game_session']
            aggregated_game_sessions['assessment_title'] = assessment['title']
            aggregated_game_sessions['assessment_world'] = assessment['world']

            # Add missing event_codes
            for missing_event_code in (set(event_codes) - set(present_event_codes)):
                aggregated_game_sessions[missing_event_code] = 0

            game_sessions.append(aggregated_game_sessions)

            # if idx == 100:
            #     break

            if idx % 100 == 0:
                print(f'Row {idx + 1}/{total_assessments} done')

        df_final = pd.concat(game_sessions, ignore_index=True, sort=False)
        df_final.fillna(0.0, inplace=True)
        df_final = df_final.merge(df_event_data_props_per_game_session, how='left', on=['game_session'])
        df_final = pd.get_dummies(df_final, columns=['title', 'type', 'world'], prefix='cnt')

        aggregate_columns = {
            'game_time': ['sum', 'mean', 'max', 'std', 'median'],
            'event_count': ['sum', 'mean', 'max', 'std', 'median'],
            'correct_attempts': ['sum', 'mean', 'max', 'std', 'median'],
            'uncorrect_attempts': ['sum', 'mean', 'max', 'std', 'median'],
            'accuracy_rate': ['sum', 'mean', 'max', 'std', 'median'],

            **{column: 'first'
               for column in df_final.columns if
               column.startswith('assessment') and column != 'assessment_game_session'},

            **{column: ['sum', 'mean', 'max', 'std', 'median']
               for column in df_final.columns if column.startswith('event_data_prop')},

            # Sum dummy columns (cnt_train_*, cnt_type_*, cnt_world_*, cnt_event_code_*)
            **{column: 'sum'
               for column in df_final.columns if column.startswith('cnt')}
        }

        df_final_aggregated = df_final[
            ['installation_id', 'assessment_game_session', *aggregate_columns.keys()]
        ].groupby(['installation_id', 'assessment_game_session']).agg(aggregate_columns).reset_index()
        df_final_aggregated.fillna(0.0, inplace=True)
        df_final_aggregated.columns = [_get_aggregated_column_name(column) for column in df_final_aggregated.columns]

        df_final_accuracy_groups = df_final[
            ['installation_id', 'assessment_game_session', 'accuracy_group']
        ].groupby(['installation_id', 'assessment_game_session'])

        # Convert accuracy groups to columns (accuracy_group_*), with their counts as values
        accuracy_groups = []
        for _, game_session_group in df_final_accuracy_groups:
            first_row = game_session_group.iloc[0]
            accuracy_groups.append({
                'installation_id': first_row['installation_id'],
                'assessment_game_session': first_row['assessment_game_session'],
                **_get_default_accuracy_groups_dict(),
                **game_session_group['accuracy_group'].value_counts()
            })

        df_final_accuracy_groups = pd.DataFrame(accuracy_groups).rename({
            0: 'accuracy_group_0', 1: 'accuracy_group_1', 2: 'accuracy_group_2', 3: 'accuracy_group_3',
        }, axis=1)

        df_final = df_final_aggregated.merge(
            df_final_accuracy_groups,
            on=['installation_id', 'assessment_game_session']
        )

        features_to_drop = set(df_final.columns) & unimportant_features

        print('Writing features...')
        df_final.drop(columns=features_to_drop).to_csv(f'preprocessed-data/{prefix}_features.csv', index=False)

    # Empirically (:D) determined unimportant or highly correlated features
    with open(CORRELATED_FEATURES_JSON, encoding='utf-8') as f:
        correlated_features_to_drop = json.load(f)['features']

    feature_importances = pd.read_csv(FEATURE_IMPORTANCES_CSV)
    unimportant_features = \
        set(feature_importances[feature_importances['importance'] == 0.0]['feature']).union(correlated_features_to_drop)

    df_train_labels = pd.read_csv(TRAIN_LABELS_CSV)
    df_train = pd.read_csv('preprocessed-data/train_preprocessed.csv')
    df_train_event_data_props_per_game_session = pd.read_csv('preprocessed-data/' +
                                                             'train_event_data_props_per_game_session.csv')
    event_codes = list(pd.read_csv('preprocessed-data/event_codes.csv')['event_code'])
    event_codes = ['cnt_event_code_' + str(column) for column in event_codes]

    title_to_type_and_world = df_train[['title', 'world']].groupby('title').agg('first').apply(list).to_dict()
    # Add assessment world to the title
    df_train_labels['world'] = df_train_labels.apply(
        lambda assessment: title_to_type_and_world['world'][assessment['title']],
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

    _compute_features(df_train, 'train', df_train_labels, df_train_event_data_props_per_game_session)

    del df_train
    del df_train_event_data_props_per_game_session
    gc.collect()

    print('Preparing test data...')
    df_test = pd.read_csv('preprocessed-data/test_preprocessed.csv')
    df_test_event_data_props_per_game_session = pd.read_csv('preprocessed-data/' +
                                                            'test_event_data_props_per_game_session.csv')

    df_test_assessments_to_predict = df_test.groupby(['installation_id']).last().reset_index()
    df_test_assessments_to_predict['assessment_most_common_title_accuracy_group'] = \
        df_test_assessments_to_predict['title'].map(title_to_acc_group_dict)

    _add_time_features(df_test_assessments_to_predict)
    df_test['timestamp'] = (pd.to_datetime(df_test['timestamp']).astype(int) / 10 ** 9).astype(int)

    _compute_features(
        df_test,
        'test',
        df_test_assessments_to_predict,
        df_test_event_data_props_per_game_session
    )

    del df_test
    del df_test_event_data_props_per_game_session
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
        # n_splits = 8
        n_splits = 5
        kf = GroupKFold(n_splits=n_splits)

        scores = []
        y_test = np.zeros((X_test.shape[0], 4))
        for fold, (train_split, test_split) in enumerate(kf.split(X, y_target, train_installation_ids)):
            print(f'Starting fold {fold}...')

            x_train, x_val, y_train, y_val = X[train_split], X[test_split], y_target[train_split], y_target[
                test_split]

            # params = {
            #     'colsample_bytree': 0.5623023061457832,
            #     'learning_rate': 0.038693355253367305,
            #     'max_depth': 9,
            #     'min_child_samples': 362,
            #     'min_child_weight': 170.0816404635047,
            #     'num_leaves': 684,
            #     'reg_alpha': 1.1546764870218185,
            #     'reg_lambda': 3.4950993744727974,
            #     'subsample': 0.27575842493928343
            # }
            params = {"colsample_bytree": 0.2933030554201025, "learning_rate": 0.08759660076786627,
                      "max_depth": 19, "min_child_samples": 401,
                      "min_child_weight": 463.5285162107799,
                      "num_leaves": 359.26563150569143, "reg_alpha": 5.955140364479005, "reg_lambda": 5.629012060716021,
                      "subsample": 0.2224423054862252}

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
