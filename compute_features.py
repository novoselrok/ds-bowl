import pandas as pd
import numpy as np


def get_default_accuracy_groups_dict():
    return {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }


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


def compute_features(df_data, df_assessments):
    game_sessions = []

    df_assessments = df_assessments.sort_values(by='installation_id')
    total_assessments = df_assessments.shape[0]

    title_to_type_and_world = df_data[['title', 'type', 'world']].groupby('title').agg('first').apply(list).to_dict()
    installation_id_to_events = dict(tuple(df_data.groupby('installation_id')))

    for idx, assessment in df_assessments.iterrows():
        installation_id = assessment['installation_id']
        start_timestamp = assessment['timestamp']

        events_for_installation_id = installation_id_to_events[installation_id]

        previous_events: pd.DataFrame = events_for_installation_id[
            events_for_installation_id['timestamp'] < start_timestamp]

        assessment_type = title_to_type_and_world['type'][assessment['title']]
        assessment_world = title_to_type_and_world['world'][assessment['title']]

        if previous_events.shape[0] == 0:
            game_sessions.append(
                pd.DataFrame({
                    'game_time': 0,
                    'event_count': 0,
                    'correct_attempts': 0,
                    'uncorrect_attempts': 0,
                    'accuracy_group': np.nan,
                    'title': '__no_title',
                    'type': '__no_type',
                    'world': '__no_world',
                    'installation_id': installation_id,
                    'assessment_start_timestamp': start_timestamp,
                    'assessment_game_session': assessment['game_session'],
                    'assessment_title': assessment['title'],
                    'assessment_type': assessment_type,
                    'assessment_world': assessment_world,
                }, index=[0])
            )
            continue

        # TODO: Features
        # - Get game time and event count for all game sessions
        # - Compute accuracy groups and sum them for all game sessions
        # - Count previous different titles, worlds and types
        # - use current assessment title, world and type as categorical variable
        # - add start_timestamp for later sorting and use as a feature (dayofweek, etc)
        # TODO: Figure out how to add event_code
        # TODO: Time from previous assessment
        # TODO: Index of the current assessment (user might do worse on first assessment than on later assessments)

        aggregated_game_sessions = previous_events[
            ['game_session', 'game_time', 'event_count', 'is_correct_attempt', 'is_uncorrect_attempt', 'title', 'type',
             'world']
        ].groupby('game_session').agg(
            {'game_time': 'max', 'event_count': 'max', 'is_correct_attempt': 'sum', 'is_uncorrect_attempt': 'sum',
             'title': 'first', 'type': 'first', 'world': 'first'}
        ).reset_index()

        aggregated_game_sessions['accuracy_group'] = aggregated_game_sessions \
            .apply(compute_accuracy_group, axis=1)

        # Everything with the assessment_* prefix is related to
        # the assessment we are predicting the accuracy_group for.
        # Everything else is aggregated from game sessions that happened before the assessment.
        aggregated_game_sessions['installation_id'] = installation_id
        aggregated_game_sessions['assessment_game_session'] = assessment['game_session']
        aggregated_game_sessions['assessment_start_timestamp'] = start_timestamp
        aggregated_game_sessions['assessment_title'] = assessment['title']
        aggregated_game_sessions['assessment_type'] = assessment_type
        aggregated_game_sessions['assessment_world'] = assessment_world

        aggregated_game_sessions.rename({
            'is_correct_attempt': 'correct_attempts',
            'is_uncorrect_attempt': 'uncorrect_attempts',
        }, axis=1, inplace=True)

        game_sessions.append(aggregated_game_sessions)

        # if idx == 500:
        #     break

        if idx % 100 == 0:
            print(f'Row {idx + 1}/{total_assessments} done')

    df_final = pd.concat(game_sessions, ignore_index=True, sort=False)
    df_final = pd.get_dummies(df_final, columns=['title', 'type', 'world'], prefix='cnt')

    aggregate_columns = {
        'game_time': ['sum', 'mean', 'max'],
        'event_count': ['sum', 'mean', 'max'],
        'correct_attempts': 'sum',
        'uncorrect_attempts': 'sum',
        'assessment_start_timestamp': 'first',
        'assessment_title': 'first',
        'assessment_type': 'first',
        'assessment_world': 'first',
        # Sum dummy columns (cnt_train_*, cnt_type_*, cnt_world_*)
        **{column: 'sum' for column in df_final.columns if column.startswith('cnt')}
    }

    df_final_aggregated = df_final[
        ['installation_id', 'assessment_game_session', *aggregate_columns.keys()]
    ].groupby(['installation_id', 'assessment_game_session']).agg(aggregate_columns).reset_index()
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

    df_final = df_final_aggregated.merge(df_final_accuracy_groups, on=['installation_id', 'assessment_game_session'])

    return df_final


df_train_labels = pd.read_csv('data/train_labels.csv')
df_train = pd.read_csv('preprocessed-data/train.csv')
df_test = pd.read_csv('preprocessed-data/test.csv')

# Convert timestamps
df_train['timestamp'] = (pd.to_datetime(df_train['timestamp']).astype(int) / 10 ** 9).astype(int)
df_test['timestamp'] = (pd.to_datetime(df_test['timestamp']).astype(int) / 10 ** 9).astype(int)

print('Preparing train data...')
# Add game session start timestamp to train labels
game_session_start_timestamps = df_train.groupby(
    ['installation_id', 'game_session']
).first()['timestamp'].reset_index()

df_train_labels = df_train_labels.merge(
    game_session_start_timestamps, on=['installation_id', 'game_session'], how='left'
)
train_data = compute_features(df_train, df_train_labels)

print('Preparing test data...')
test_assessments_to_predict = df_test.groupby(['installation_id']).last().reset_index()
test_data = compute_features(df_test, test_assessments_to_predict)

train_columns = set(train_data.columns.tolist())
test_columns = set(test_data.columns.tolist())
if train_columns != test_columns:
    print('Mismatch between train and test columns, fixing...')
    train_data = train_data.drop(columns=list(train_columns - test_columns))
    test_data = test_data.drop(columns=list(test_columns - train_columns))

print('Writing files...')
train_data.to_csv('preprocessed-data/train_features.csv', index=False)
test_data.to_csv('preprocessed-data/test_features.csv', index=False)
