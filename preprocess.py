import json
import os

import pandas as pd
import numpy as np

with open('event_props.json', encoding='utf-8') as f:
    event_ids_to_props = json.load(f)


def extract_props_from_event_data(df_events):
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


def preprocess_data(df_data, output_path, event_data_props_per_game_session_path):
    is_assessment = df_data['type'] == 'Assessment'

    is_bird_measurer_attempt = (df_data['title'] == 'Bird Measurer (Assessment)') & (df_data['event_code'] == 4110)
    is_non_bird_measurer_attempt = (df_data['title'] != 'Bird Measurer (Assessment)') & (df_data['event_code'] == 4100)
    is_assessment_attempt = is_assessment & (is_bird_measurer_attempt | is_non_bird_measurer_attempt)

    is_correct_attempt = df_data['event_data'].str.contains('"correct":true')
    is_uncorrect_attempt = df_data['event_data'].str.contains('"correct":false')

    df_data['is_correct_attempt'] = (is_assessment_attempt & is_correct_attempt).astype(int)
    df_data['is_uncorrect_attempt'] = (is_assessment_attempt & is_uncorrect_attempt).astype(int)

    extract_props_from_event_data(df_data[['game_session', 'event_id', 'event_data']]) \
        .to_csv(event_data_props_per_game_session_path, index=False)

    df_data = df_data.drop(columns=['event_data'])
    df_data.to_csv(output_path, index=False)


os.makedirs('preprocessed-data', exist_ok=True)

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
df_train_labels = pd.read_csv('data/train_labels.csv')

# Only keep installation ids from train labels
installation_ids = df_train_labels['installation_id'].unique()
df_train = df_train[df_train['installation_id'].isin(installation_ids)].reset_index(drop=True)

event_codes = set(df_train['event_code'].unique()) & set(df_test['event_code'].unique())
pd.DataFrame({'event_code': list(event_codes)}).to_csv('preprocessed-data/event_codes.csv', index=False)

print('Preprocessing train...')
preprocess_data(
    df_train,
    'preprocessed-data/train.csv',
    'preprocessed-data/train_event_data_props_per_game_session.csv'
)

print('Preprocessing test...')
preprocess_data(df_test, 'preprocessed-data/test.csv', 'preprocessed-data/test_event_data_props_per_game_session.csv')
