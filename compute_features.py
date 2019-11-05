import pandas as pd
import numpy as np
import gc


def add_time_features(df):
    timestamp = pd.to_datetime(df['timestamp'])
    df['timestamp'] = (timestamp.astype(int) / 10 ** 9).astype(int)
    df['assessment_hour'] = timestamp.dt.hour
    df['assessment_dayofweek'] = timestamp.dt.dayofweek


def get_default_accuracy_groups_dict():
    return {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }


def compute_accuracy_rate(row):
    correct, uncorrect = row['is_correct_attempt'], row['is_uncorrect_attempt']
    if correct == 0 and uncorrect == 0:
        return 0.0

    return correct / float(correct + uncorrect)


def get_aggregated_column_name(column):
    if not isinstance(column, tuple):
        return column

    if column[1] == 'first' or column[1] == '':
        return column[0]

    return '_'.join(column)


def compute_accuracy_group(row):
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


def get_pivoted_game_time_and_event_count_per_column(df, column):
    """
    Data structure
    installation_id -> sum/game_time/Dino Dive, mean/event_code_123/Dino Drink, etc..
    """
    pivoted = df[['installation_id', 'assessment_game_session', column, 'game_time', 'event_count']] \
        .pivot_table(
        index=['installation_id', 'assessment_game_session'],
        columns=[column],  # title, type, world
        values=['game_time', 'event_count'],
        aggfunc=[np.sum, np.mean, np.max],
    )
    pivoted.fillna(0, inplace=True)
    pivoted.columns = [get_aggregated_column_name(column) for column in list(pivoted.columns)]
    return pivoted


def compute_features(df_data, df_assessments, df_event_data_props_per_game_session, event_codes):
    game_sessions = []

    event_codes = ['cnt_event_code_' + str(column) for column in event_codes]
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

        assessment_world = assessment['world']

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
                    'assessment_start_timestamp': start_timestamp,
                    'assessment_game_session': assessment['game_session'],
                    'assessment_title': assessment['title'],
                    'assessment_world': assessment_world,
                    'assessment_most_common_title_accuracy_group': assessment[
                        'assessment_most_common_title_accuracy_group'],
                    'assessment_dayofweek': assessment['assessment_dayofweek'],
                    'assessment_hour': assessment['assessment_hour'],
                    **{column: 0 for column in event_codes}
                }, index=[0])
            )
            continue

        aggregated_game_sessions = previous_events[
            ['game_session', 'game_time', 'event_count', 'is_correct_attempt', 'is_uncorrect_attempt', 'title', 'type',
             'world', *present_event_codes]
        ].groupby('game_session').agg(
            {'game_time': 'max', 'event_count': 'max', 'is_correct_attempt': 'sum', 'is_uncorrect_attempt': 'sum',
             'title': 'first', 'type': 'first', 'world': 'first', **{column: 'sum' for column in present_event_codes}}
        ).reset_index()

        aggregated_game_sessions['accuracy_group'] = aggregated_game_sessions \
            .apply(compute_accuracy_group, axis=1)

        aggregated_game_sessions['accuracy_rate'] = aggregated_game_sessions \
            .apply(compute_accuracy_rate, axis=1)

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
        aggregated_game_sessions['assessment_start_timestamp'] = start_timestamp
        aggregated_game_sessions['assessment_title'] = assessment['title']
        aggregated_game_sessions['assessment_world'] = assessment_world

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
           for column in df_final.columns if column.startswith('assessment') and column != 'assessment_game_session'},

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
    df_final_aggregated.columns = [get_aggregated_column_name(column) for column in df_final_aggregated.columns]

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
            **get_default_accuracy_groups_dict(),
            **game_session_group['accuracy_group'].value_counts()
        })

    df_final_accuracy_groups = pd.DataFrame(accuracy_groups).rename({
        0: 'accuracy_group_0',
        1: 'accuracy_group_1',
        2: 'accuracy_group_2',
        3: 'accuracy_group_3',
    }, axis=1)

    df_final = df_final_aggregated.merge(
        df_final_accuracy_groups,
        on=['installation_id', 'assessment_game_session']
    )

    return df_final


df_train_labels = pd.read_csv('data/train_labels.csv')
df_train = pd.read_csv('preprocessed-data/train.csv')
df_train_event_data_props_per_game_session = pd.read_csv('preprocessed-data/train_event_data_props_per_game_session.csv')
event_codes = list(pd.read_csv('preprocessed-data/event_codes.csv')['event_code'])

title_to_type_and_world = df_train[['title', 'type', 'world']].groupby('title').agg('first').apply(list).to_dict()
df_train_labels['type'] = df_train_labels.apply(
    lambda assessment: title_to_type_and_world['type'][assessment['title']],
    axis=1
)
df_train_labels['world'] = df_train_labels.apply(
    lambda assessment: title_to_type_and_world['world'][assessment['title']],
    axis=1
)

print('Preparing train data...')

title_to_acc_group_dict = dict(
    df_train_labels.groupby('title')['accuracy_group'].agg(lambda x: x.value_counts().index[0])
)
df_train_labels['assessment_most_common_title_accuracy_group'] = df_train_labels['title'].map(title_to_acc_group_dict)

# Add game session start timestamp to train labels
game_session_start_timestamps = df_train.groupby(
    ['installation_id', 'game_session']
).first()['timestamp'].reset_index()

df_train_labels = df_train_labels.merge(
    game_session_start_timestamps, on=['installation_id', 'game_session'], how='left'
)

add_time_features(df_train_labels)

df_train['timestamp'] = (pd.to_datetime(df_train['timestamp']).astype(int) / 10 ** 9).astype(int)
train_data = compute_features(df_train, df_train_labels, df_train_event_data_props_per_game_session, event_codes)

print('Writing files...')
train_data.to_csv('preprocessed-data/train_features.csv', index=False)

del df_train
del train_data
del df_train_event_data_props_per_game_session
gc.collect()

print('Preparing test data...')
df_test = pd.read_csv('preprocessed-data/test.csv')
df_test_event_data_props_per_game_session = pd.read_csv('preprocessed-data/test_event_data_props_per_game_session.csv')
test_assessments_to_predict = df_test.groupby(['installation_id']).last().reset_index()
test_assessments_to_predict['assessment_most_common_title_accuracy_group'] = test_assessments_to_predict['title'].map(
    title_to_acc_group_dict)

add_time_features(test_assessments_to_predict)

df_test['timestamp'] = (pd.to_datetime(df_test['timestamp']).astype(int) / 10 ** 9).astype(int)
test_data = compute_features(df_test, test_assessments_to_predict, df_test_event_data_props_per_game_session, event_codes)

print('Writing files...')
test_data.to_csv('preprocessed-data/test_features.csv', index=False)

del df_test
del test_data
del df_test_event_data_props_per_game_session
gc.collect()
