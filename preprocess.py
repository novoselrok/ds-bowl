import json
import os

import pandas as pd

with open('event_props.json', encoding='utf-8') as f:
    event_ids_to_props = json.load(f)


def extract_props_from_event_data(df_events):
    extracted_props = []
    for idx, event in df_events.iterrows():
        event_id = event['event_id']
        event_data = json.loads(event['event_data'])

        if event_id not in event_ids_to_props:
            extracted_props.append({})
            continue

        props = event_ids_to_props[event_id]
        values = {f'event_data_prop_{event_id}_{prop}': event_data[prop] for prop in props}
        extracted_props.append(values)

        if idx % 10000 == 0:
            print(f'Row {idx + 1}/{df_events.shape[0]} done')

    df_event_data_props = pd.SparseDataFrame(extracted_props)
    df_event_data_props.fillna(0, inplace=True)
    return df_event_data_props


def preprocess_data(df_data, output_path):
    is_assessment = df_data['type'] == 'Assessment'

    is_bird_measurer_attempt = (df_data['title'] == 'Bird Measurer (Assessment)') & (df_data['event_code'] == 4110)
    is_non_bird_measurer_attempt = (df_data['title'] != 'Bird Measurer (Assessment)') & (df_data['event_code'] == 4100)
    is_assessment_attempt = is_assessment & (is_bird_measurer_attempt | is_non_bird_measurer_attempt)

    is_correct_attempt = df_data['event_data'].str.contains('"correct":true')
    is_uncorrect_attempt = df_data['event_data'].str.contains('"correct":false')

    df_data['is_correct_attempt'] = (is_assessment_attempt & is_correct_attempt).astype(int)
    df_data['is_uncorrect_attempt'] = (is_assessment_attempt & is_uncorrect_attempt).astype(int)

    # df_event_data_props = extract_props_from_event_data(df_data[['event_id', 'event_data']])

    df_data = df_data.drop(columns=['event_data'])
    # df_data = df_data.merge(df_event_data_props, left_index=True, right_index=True)
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
preprocess_data(df_train, 'preprocessed-data/train.csv')
print('Preprocessing test...')
preprocess_data(df_test, 'preprocessed-data/test.csv')
